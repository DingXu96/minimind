# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,  # Dropout概率，用于防止过拟合
            bos_token_id: int = 1,  # 序列开始标记的ID
            eos_token_id: int = 2,  # 序列结束标记的ID
            hidden_act: str = 'silu',  # 隐藏层激活函数类型
            hidden_size: int = 512,  # 隐藏层维度大小
            intermediate_size: int = None,  # 中间层维度大小，如果为None则会自动计算
            max_position_embeddings: int = 32768,  # 最大位置编码长度
            num_attention_heads: int = 8,  # 注意力头的数量
            num_hidden_layers: int = 8,  # 隐藏层的数量
            num_key_value_heads: int = 2,  # 键值头的数量，用于分组查询注意力
            vocab_size: int = 6400,  # 词汇表大小
            rms_norm_eps: float = 1e-05,  # RMS归一化层的epsilon值，用于数值稳定性
            rope_theta: int = 1000000.0,  # RoPE位置编码的theta参数
            flash_attn: bool = True,  # 是否使用Flash Attention优化
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,  # 是否使用MoE (Mixture of Experts) 结构
            num_experts_per_tok: int = 2,  # 每个token选择的专家数量
            n_routed_experts: int = 4,  # 总的专家数量
            n_shared_experts: int = 1,  # 共享专家数量
            scoring_func: str = 'softmax',  # 评分函数，默认为'softmax'
            aux_loss_alpha: float = 0.1,  # 辅助损失的alpha参数，用于平衡辅助损失的权重
            seq_aux: bool = True,  # 是否在序列级别上计算辅助损失
            norm_topk_prob: bool = True,  # 是否标准化top-k概率
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout  # 设置dropout概率
        self.bos_token_id = bos_token_id  # 设置序列开始标记ID
        self.eos_token_id = eos_token_id  # 设置序列结束标记ID
        self.hidden_act = hidden_act  # 设置隐藏层激活函数
        self.hidden_size = hidden_size  # 设置隐藏层维度
        self.intermediate_size = intermediate_size  # 设置中间层维度
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码长度
        self.num_attention_heads = num_attention_heads  # 设置注意力头数量
        self.num_hidden_layers = num_hidden_layers  # 设置隐藏层数量
        self.num_key_value_heads = num_key_value_heads  # 设置键值头数量
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.rms_norm_eps = rms_norm_eps  # 设置RMS归一化epsilon值
        self.rope_theta = rope_theta  # 设置RoPE位置编码theta参数
        self.flash_attn = flash_attn  # 设置是否使用Flash Attention
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe  # 设置是否使用MoE结构
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # epsilon值，用于数值稳定性
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def _norm(self, x):
        # 计算RMS归一化值: x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 应用RMS归一化并使用可学习参数进行缩放
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    # 预计算旋转位置编码的频率值
    # dim: 每个注意力头的维度
    # end: 最大序列长度
    # theta: 频率计算参数
    
    # 计算每个维度的频率值
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 创建位置索引
    t = torch.arange(end, device=freqs.device)
    # 计算每个位置和每个维度的频率
    freqs = torch.outer(t, freqs).float()
    # 计算余弦和正弦值，用于旋转位置编码
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 应用旋转位置编码到查询和键向量
    # q: 查询向量 [batch_size, seq_len, num_heads, head_dim]
    # k: 键向量 [batch_size, seq_len, num_heads, head_dim]
    # cos, sin: 预计算的余弦和正弦值
    # position_ids: 位置索引（可选）
    # unsqueeze_dim: 扩展维度的位置
    
    def rotate_half(x):
        # 将向量的后半部分取负并交换前后两部分
        # 这是旋转位置编码的核心操作
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 应用旋转位置编码公式:
    # x_new = x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值对张量以匹配查询头的数量
    这在分组查询注意力(GQA)中使用，其中键值头数量少于查询头数量
    
    Args:
        x: 输入张量，形状为 [batch_size, seq_len, num_key_value_heads, head_dim]
        n_rep: 重复次数，通常是查询头数量/键值头数量
    
    Returns:
        重复后的张量，形状为 [batch_size, seq_len, num_key_value_heads * n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        # 如果不需要重复，直接返回原张量
        return x
    # 通过扩展和重塑操作重复键值对
    return (
        x[:, :, :, None, :]  # 在键值头维度后添加新维度
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)  # 扩展新维度
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)  # 重塑为最终形状
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 设置键值头数量，如果未指定则默认等于注意力头数量
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # 确保注意力头数量能被键值头数量整除
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads  # 本地查询头数量
        self.n_local_kv_heads = self.num_key_value_heads  # 本地键值头数量
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 重复次数，用于分组查询注意力
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个注意力头的维度
        
        # 定义线性投影层
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)  # 查询投影
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 键投影
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)  # 值投影
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)  # 输出投影
        
        # 定义dropout层
        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接dropout
        self.dropout = args.dropout  # dropout概率
        
        # 检查是否可以使用Flash Attention优化
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 接收预计算的cos和sin位置编码
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 用于KV缓存的过去键值
                use_cache=False,  # 是否使用KV缓存
                attention_mask: Optional[torch.Tensor] = None):  # 注意力掩码
        bsz, seq_len, _ = x.shape
        # 通过线性投影计算查询、键、值
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 重塑张量形状以匹配注意力头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 获取位置编码
        cos, sin = position_embeddings
        # 应用旋转位置编码到查询和键
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # KV缓存实现：如果存在过去的键值，则与当前键值拼接
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 如果使用缓存，则保存当前键值作为过去的键值
        past_kv = (xk, xv) if use_cache else None

        # 转置张量以匹配注意力计算的格式，并重复键值以匹配查询头数量
        xq, xk, xv = (
            xq.transpose(1, 2),  # [batch_size, num_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),  # 重复键以匹配查询头数量
            repeat_kv(xv, self.n_rep).transpose(1, 2)   # 重复值以匹配查询头数量
        )

        # 使用Flash Attention或标准注意力计算
        if self.flash and seq_len != 1:
            # 使用Flash Attention优化
            dropout_p = self.dropout if self.training else 0.0  # 训练时使用dropout
            attn_mask = None
            # 如果提供了注意力掩码，则处理为合适的格式
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            # 使用PyTorch内置的缩放点积注意力函数
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            # 使用标准注意力计算
            # 计算注意力分数
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 应用因果掩码（防止未来信息泄露）
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            # 如果提供了注意力掩码，则应用到注意力分数
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # 应用softmax归一化并进行dropout
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # 计算注意力输出
            output = scores @ xv

        # 重塑输出张量并应用输出投影和残差dropout
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 如果未指定中间层大小，则根据隐藏层大小自动计算
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)  # 计算中间层大小
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # 向上取整到64的倍数
        
        # 定义前馈网络的三个线性投影层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 门控投影
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 下投影
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 上投影
        
        self.dropout = nn.Dropout(config.dropout)  # Dropout层
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数

    def forward(self, x):
        # 前馈网络的计算过程：
        # 1. 通过门控投影和上投影分别计算两个分支
        # 2. 对门控投影的结果应用激活函数
        # 3. 将激活后的门控结果与上投影结果相乘
        # 4. 通过下投影层映射回原始维度
        # 5. 应用dropout
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 总的专家数量

        self.scoring_func = config.scoring_func  # 评分函数
        self.alpha = config.aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = config.seq_aux  # 是否在序列级别上计算辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否标准化top-k概率
        self.gating_dim = config.hidden_size  # 门控维度
        # 门控权重矩阵，形状为[专家数量, 隐藏层维度]
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self) -> None:
        # 使用Kaiming均匀初始化门控权重
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # MoE门控机制的前向传播
        bsz, seq_len, h = hidden_states.shape
        # 重塑隐藏状态以进行线性变换
        hidden_states = hidden_states.view(-1, h)
        # 计算每个token对每个专家的logits
        logits = F.linear(hidden_states, self.weight, None)
        
        # 根据评分函数计算分数
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 使用softmax计算分数
        else:
            raise NotImplementedError(f'不支持的MoE门控评分函数: {self.scoring_func}')

        # 选择top-k个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果top-k大于1且需要标准化概率，则进行标准化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 计算辅助损失（用于训练时平衡专家负载）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                # 在序列级别计算辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 在token级别计算辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0  # 不计算辅助损失
        
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 创建专家模块列表，每个专家都是一个前馈网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 创建门控机制
        self.gate = MoEGate(config)
        # 如果有共享专家，则创建共享专家模块列表
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        # MoE前馈网络的前向传播
        identity = x  # 保存残差连接的输入
        orig_shape = x.shape  # 保存原始形状
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])  # 重塑输入张量
        flat_topk_idx = topk_idx.view(-1)  # 展平专家索引
        
        # 根据训练或推理模式选择不同的计算方式
        if self.training:
            # 训练模式下的计算
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)  # 重复输入以匹配专家数量
            y = torch.empty_like(x, dtype=torch.float16)  # 创建输出张量
            # 为每个专家计算输出
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            # 根据专家权重组合输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)  # 恢复原始形状
        else:
            # 推理模式下的计算
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 如果有共享专家，则添加共享专家的输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        self.aux_loss = aux_loss  # 保存辅助损失
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # MoE推理模式下的专家选择和计算
        expert_cache = torch.zeros_like(x)  # 创建专家输出缓存
        idxs = flat_expert_indices.argsort()  # 对专家索引进行排序
        # 计算每个专家处理的token数量累积和
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok  # 计算token索引
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]  # 计算起始索引
            if start_idx == end_idx:  # 如果没有token分配给这个专家，则跳过
                continue
            expert = self.experts[i]  # 获取对应专家
            exp_token_idx = token_idxs[start_idx:end_idx]  # 获取分配给该专家的token索引
            expert_tokens = x[exp_token_idx]  # 获取分配给该专家的token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)  # 计算专家输出
            # 根据专家权重调整输出
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将专家输出累加到缓存中
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads  # 注意力头数量
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.head_dim = config.hidden_size // config.num_attention_heads  # 每个注意力头的维度
        self.self_attn = Attention(config)  # 自注意力机制

        self.layer_id = layer_id  # 层ID
        # 输入层归一化和注意力后归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 前馈网络，根据配置选择普通前馈网络或MoE前馈网络
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # Transformer块的前向传播
        residual = hidden_states  # 保存残差连接的输入
        # 自注意力计算
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual  # 残差连接
        
        residual = hidden_states  # 保存前馈网络的残差连接输入
        # 前馈网络计算
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states += residual  # 残差连接
        
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)  # Dropout层
        # 创建Transformer块列表
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算旋转位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        # 注册为缓冲区，不参与梯度更新
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        # 模型前向传播
        batch_size, seq_length = input_ids.shape
        # 初始化past_key_values，如果未提供则创建全为None的列表
        past_key_values = past_key_values or [None] * len(self.layers)
        # 计算起始位置，用于处理缓存
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词嵌入和dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前序列的位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []  # 用于存储每层的KV缓存
        # 逐层计算Transformer块
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)  # 保存KV缓存

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # 计算MoE的辅助损失
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig  # 配置类

    def __init__(self, config: MiniMindConfig = None):
        # 初始化因果语言模型
        self.config = config or MiniMindConfig()  # 配置对象
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)  # MiniMind主模型
        # 语言模型头部，将隐藏状态映射到词汇表
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 共享嵌入层和输出层权重
        self.model.embed_tokens.weight = self.lm_head.weight
        # 输出对象
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        # 前向传播函数
        # 调用主模型进行计算
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 确定需要保留的logits索引
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 通过语言模型头部计算logits
        logits = self.lm_head(h[:, slice_indices, :])
        # 设置输出对象的各个属性
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
