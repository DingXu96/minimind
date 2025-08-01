# 导入必要的库
import os
import sys
# 设置包名，确保可以正确导入trainer包中的模块
__package__ = "trainer"
# 将项目根目录添加到Python路径中，以便能够导入项目中的其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse  # 用于解析命令行参数
import time      # 用于计时
import math      # 数学计算库
import warnings  # 警告处理
import torch     # PyTorch深度学习框架
import torch.distributed as dist  # PyTorch分布式训练支持
from torch import optim, nn  # PyTorch优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载和分布式采样
from contextlib import nullcontext  # 上下文管理器
from transformers import AutoTokenizer  # HuggingFace的自动分词器
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # 自定义的MiniMind模型
from dataset.lm_dataset import PretrainDataset  # 自定义的预训练数据集

# 忽略警告信息，避免输出过多警告
warnings.filterwarnings('ignore')


def Logger(content):
    """
    日志打印函数，只在主进程中打印日志信息
    在分布式训练中，避免所有进程都打印相同信息
    
    Args:
        content: 要打印的内容
    """
    # 如果不是分布式训练或者当前是主进程（rank=0），则打印日志
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    学习率调度函数，使用余弦退火学习率调度器
    
    Args:
        current_step: 当前训练步数
        total_steps: 总训练步数
        lr: 基础学习率
    
    Returns:
        调整后的学习率
    """
    # 使用余弦退火公式计算学习率
    # 初始学习率为lr/10 + 0.5*lr，然后逐渐衰减到lr/10
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    训练一个epoch的函数
    
    Args:
        epoch: 当前epoch数
        wandb: Weights & Biases对象，用于记录训练日志
    """
    # 定义损失函数，使用交叉熵损失，reduction='none'表示不对损失求平均
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    # 记录训练开始时间
    start_time = time.time()
    
    # 遍历训练数据加载器
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据移动到指定设备（GPU/CPU）
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 计算当前学习率
        # current_step = epoch * iter_per_epoch + step
        # total_steps = args.epochs * iter_per_epoch
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用自动混合精度训练上下文
        with ctx:
            # 前向传播
            res = model(X)
            # 计算损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # 将logits展平为二维张量
                Y.view(-1)  # 将标签展平为一维张量
            ).view(Y.size())  # 恢复为原始标签的形状
            
            # 使用loss_mask对损失进行加权平均
            # loss_mask用于忽略填充位置的损失
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 如果使用MoE（Mixture of Experts），加上辅助损失
            loss += res.aux_loss
            
            # 梯度累积：将损失除以累积步数
            loss = loss / args.accumulation_steps

        # 反向传播，使用梯度缩放器处理混合精度训练
        scaler.scale(loss).backward()

        # 梯度累积：每隔accumulation_steps步更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            # 在更新参数前进行梯度缩放的逆操作
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新模型参数
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度
            optimizer.zero_grad(set_to_none=True)

        # 定期打印训练日志
        if step % args.log_interval == 0:
            # 计算已用时间
            spend_time = time.time() - start_time
            # 打印训练信息
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,           # 当前epoch数
                    args.epochs,         # 总epoch数
                    step,                # 当前step
                    iter_per_epoch,      # 每个epoch的step数
                    loss.item() * args.accumulation_steps,  # 恢复原始损失值
                    optimizer.param_groups[-1]['lr'],       # 当前学习率
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))  # 预计剩余时间

            # 如果使用wandb，则记录训练日志
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # 切换到评估模式
            model.eval()
            # 根据是否使用MoE决定保存文件名
            moe_path = '_moe' if lm_config.use_moe else ''
            # 构造保存路径
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            # 根据是否使用分布式训练获取模型状态字典
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 转换为半精度（float16）以节省存储空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            # 保存模型
            torch.save(state_dict, ckp)
            # 切换回训练模式
            model.train()


def init_model(lm_config):
    """
    初始化模型和分词器
    
    Args:
        lm_config: 模型配置对象
    
    Returns:
        model: 初始化的模型
        tokenizer: 分词器
    """
    # 从预训练模型路径加载分词器
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    # 初始化模型并移动到指定设备
    model = MiniMindForCausalLM(lm_config).to(args.device)
    # 打印模型可训练参数量
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    """
    初始化分布式训练模式
    """
    # 如果不是分布式训练则直接返回
    if not ddp: 
        return
    # 声明全局变量
    global ddp_local_rank, DEVICE

    # 初始化分布式训练后端为NCCL（NVIDIA Collective Communications Library）
    dist.init_process_group(backend="nccl")
    # 获取分布式训练相关环境变量
    ddp_rank = int(os.environ["RANK"])          # 当前进程的rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 当前节点上的本地rank
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    # 设置当前设备
    DEVICE = f"cuda:{ddp_local_rank}"
    # 设置当前CUDA设备
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")  # 输出目录
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)          # 训练轮数
    parser.add_argument("--batch_size", type=int, default=32)     # 批次大小
    parser.add_argument("--learning_rate", type=float, default=5e-4)  # 学习率
    # 设备设置，如果CUDA可用则使用GPU，否则使用CPU
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 数据类型
    parser.add_argument("--use_wandb", action="store_true")       # 是否使用wandb
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")  # wandb项目名
    parser.add_argument("--num_workers", type=int, default=1)     # 数据加载器的工作进程数
    parser.add_argument("--ddp", action="store_true")             # 是否使用分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=8)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)   # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)    # 预热步数
    parser.add_argument("--log_interval", type=int, default=100)  # 日志打印间隔
    parser.add_argument("--save_interval", type=int, default=100) # 模型保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)     # 本地rank（用于分布式训练）
    parser.add_argument('--hidden_size', default=512, type=int)   # 隐藏层大小
    parser.add_argument('--num_hidden_layers', default=8, type=int)  # 隐藏层数量
    parser.add_argument('--max_seq_len', default=512, type=int)   # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)    # 是否使用MoE
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")  # 数据路径
    args = parser.parse_args()

    # 初始化模型配置
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    # 设置保存目录
    args.save_dir = os.path.join(args.out_dir)
    # 创建目录（如果不存在）
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    # 计算每个迭代的token数量
    tokens_per_iter = args.batch_size * args.max_seq_len
    # 获取设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置wandb运行名称
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置自动混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 检查是否为分布式训练
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 设置随机种子以确保结果可复现
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # 如果是分布式训练，初始化分布式模式
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        # 为每个进程设置不同的随机种子
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    # 初始化wandb（如果启用）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)
    # 初始化训练数据集
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 如果是分布式训练，创建分布式采样器
    train_sampler = DistributedSampler(train_ds) if ddp else None
    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,      # 锁页内存，提高数据传输速度
        drop_last=False,      # 不丢弃最后一个不完整的批次
        shuffle=False,        # 不打乱数据（分布式训练中由采样器处理）
        num_workers=args.num_workers,  # 工作进程数
        sampler=train_sampler  # 采样器
    )

    # 初始化梯度缩放器（用于混合精度训练）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 如果是分布式训练，包装模型
    if ddp:
        # 指定在分布式训练中忽略的参数和缓冲区
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        # 使用DistributedDataParallel包装模型
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
