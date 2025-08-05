# 导入操作系统接口模块，用于处理文件路径等操作
# os模块提供了与操作系统交互的功能，如文件路径操作、目录操作等
import os
# 导入系统特定的参数和函数
# sys模块提供了访问和使用Python解释器的一些函数和变量
import sys

# 设置当前模块的包名，确保正确的包结构
# __package__用于标识当前模块所属的包，有助于正确解析相对导入
__package__ = "trainer"
# 将项目根目录添加到Python路径中，使得可以导入项目中的其他模块
# os.path.dirname(__file__)获取当前文件所在的目录
# os.path.join(os.path.dirname(__file__), '..')将当前目录与上级目录连接
# os.path.abspath(...)获取绝对路径
# sys.path.append(...)将计算出的项目根目录路径添加到Python模块搜索路径中
# 这样可以确保从项目根目录下的任何位置都能正确导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入命令行参数解析模块，用于处理训练时的参数配置
# argparse是Python标准库中用于命令行选项、参数和子命令解析的模块
import argparse
# 导入时间模块，用于计算训练时间等
# time模块提供了各种时间相关的函数，如获取当前时间、计算时间差等
import time
# 导入数学模块，用于学习率计算等数学运算
# math模块提供了各种数学函数，如三角函数、对数函数、幂函数等
import math
# 导入警告模块，用于忽略不必要的警告信息
# warnings模块用于控制警告的触发和处理
import warnings
# 导入PyTorch核心模块，用于构建和训练神经网络
# torch是PyTorch深度学习框架的核心模块，提供了张量计算和深度学习功能
import torch
# 导入PyTorch分布式训练模块，用于多GPU/多节点训练
# torch.distributed提供了分布式训练的支持，可以在多个GPU或多个节点上并行训练模型
import torch.distributed as dist
# 从contextlib导入nullcontext，用于上下文管理
# nullcontext是一个空的上下文管理器，在不需要特殊上下文时使用
# contextlib模块提供了实用的上下文管理器工具
from contextlib import nullcontext
# 从torch导入优化器和神经网络模块
# optim模块包含各种优化算法，如Adam、SGD等
# nn模块包含构建神经网络所需的各种层和函数
from torch import optim, nn
# 从torch.nn.parallel导入分布式数据并行模块
# DistributedDataParallel用于在多个GPU上并行训练模型，提高训练效率
from torch.nn.parallel import DistributedDataParallel
# 从torch.utils.data导入数据加载器和分布式采样器
# DataLoader用于批量加载数据，提供数据打乱、并行加载等功能
# DistributedSampler用于分布式训练中的数据采样，确保每个进程处理不同的数据子集
from torch.utils.data import DataLoader, DistributedSampler
# 从transformers库导入自动tokenizer和因果语言模型
# AutoTokenizer用于自动选择和加载适合的tokenizer，处理文本的编码和解码
# AutoModelForCausalLM用于自动选择和加载因果语言模型（用于文本生成任务）
from transformers import AutoTokenizer, AutoModelForCausalLM
# 从自定义model模块导入MiniMind配置和因果语言模型
# MiniMindConfig用于配置MiniMind模型的超参数，如隐藏层大小、层数等
# MiniMindForCausalLM是自定义的因果语言模型实现，用于文本生成任务
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
# 从自定义dataset模块导入SFT数据集
# SFTDataset用于监督微调(Supervised Fine-Tuning)数据集的处理，将原始数据转换为模型训练所需的格式
from dataset.lm_dataset import SFTDataset

# 忽略所有警告信息，避免训练过程中出现过多警告输出
# warnings.filterwarnings('ignore')会忽略所有警告，保持训练日志的清洁
# 只关注重要的错误信息，避免被大量的警告信息干扰
warnings.filterwarnings('ignore')


def Logger(content):
    """
    分布式训练中的日志记录函数
    只在非分布式训练或主进程中打印日志信息，避免在多GPU训练时重复输出
    
    在分布式训练中，多个进程可能会同时运行相同的代码，如果不加控制，
    每个进程都会输出相同的日志信息，导致日志重复和混乱。
    该函数通过检查是否为分布式训练以及当前进程的rank来决定是否输出日志，
    确保只有主进程(rank=0)输出日志信息，保持日志的清晰性。
    
    Args:
        content (str): 要打印的日志内容
        
    Returns:
        None: 无返回值，仅在满足条件时打印日志
    """
    # 如果不是分布式训练或者当前是主进程（rank=0），则打印日志
    # ddp是一个全局变量，标识是否启用分布式训练
    # dist.get_rank()获取当前进程的rank，主进程的rank为0
    # 在非分布式训练(ddp=False)或主进程(rank=0)时才输出日志
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算当前学习率的函数，使用余弦退火学习率调度策略
    
    学习率调度是深度学习训练中的重要技术，它通过动态调整学习率来提高模型的收敛速度和最终性能。
    余弦退火是一种常用的学习率调度策略，它模拟物理退火过程，使学习率从初始值逐渐降低到最小值，
    这种策略有助于模型在训练初期快速收敛，在训练后期精细调整参数以达到更好的性能。
    
    Args:
        current_step (int): 当前训练步数，表示已经完成的训练迭代次数
        total_steps (int): 总训练步数，表示整个训练过程的总迭代次数
        lr (float): 基础学习率，即初始设定的学习率值
    
    Returns:
        float: 计算得到的当前学习率
        返回值在[0.1*lr, lr]范围内变化，随着训练的进行逐渐从lr降低到0.1*lr
    
    该函数实现的学习率调度策略：
    1. 最小学习率为基础学习率的1/10，确保训练后期仍有足够的学习能力
    2. 使用余弦函数调整学习率，使学习率从最大值逐渐降低到最小值
       余弦函数的特点是开始下降较快，后来下降较慢，符合训练过程的需求
    """
    # 余弦退火学习率调度公式：lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * current_step / total_steps))
    # 其中lr_min = lr/10, lr_max = lr
    # 这里简化为：lr/10 + 0.5 * lr * (1 + cos(π * current_step / total_steps))
    # math.cos(math.pi * current_step / total_steps)计算余弦值
    # 随着current_step从0增加到total_steps，cos值从1降低到-1
    # 因此整个表达式的值从lr降低到lr/10
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """
    执行一个训练周期（epoch）的操作
    
    该函数是模型训练过程中的核心部分，负责在一个epoch内对模型进行训练。
    它处理特殊标记（如思考开始/结束、答案开始/结束标记），并为这些关键部分设置更高的损失权重，
    以提高模型在生成结构化输出时的准确性。
    
    Args:
        epoch (int): 当前训练的轮次编号，用于跟踪训练进度和计算学习率
        wandb: Weights & Biases 实例，用于记录训练过程中的指标（如损失和学习率）
              在分布式训练中，只有主进程会记录这些指标
    
    Returns:
        None: 无返回值，但会更新模型参数并记录训练过程中的指标
    """
    # 定义特殊标记的token ID，用于区分输入文本中的不同语义部分
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            # 识别特殊标记的位置（思考开始/结束、答案开始/结束标记）
            # 将所有特殊标记的token ID合并为一个张量
            special_token_ids = torch.tensor(
                start_of_think_ids + end_of_think_ids +
                start_of_answer_ids + end_of_answer_ids
            ).to(args.device)
            
            # 检查标签中哪些位置是特殊标记
            sp_ids = torch.isin(Y.view(-1), special_token_ids)
            
            # 对特殊标记位置增加额外的损失惩罚
            # 将损失掩码展平
            loss_mask = loss_mask.view(-1)
            # 计算原始损失掩码的总和
            loss_mask_sum = loss_mask.sum()
            # 对特殊标记位置设置更高的损失权重（10倍）
            loss_mask[sp_ids] = 10
            # 将损失掩码重新reshape为原始标签的形状
            loss_mask = loss_mask.view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask_sum
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/reason_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()



def init_model(lm_config):
    """
    初始化模型和tokenizer
    
    Args:
        lm_config: 语言模型配置对象
    
    Returns:
        tuple: (模型对象, tokenizer对象)
    """
    # 从预训练模型路径加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model')
    
    # 根据配置创建MiniMind因果语言模型实例
    model = MiniMindForCausalLM(lm_config)
    
    # 构建模型检查点文件路径
    moe_path = '_moe' if lm_config.use_moe else ''  # 如果使用MoE则添加后缀
    ckp = f'{args.save_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'
    
    # 从检查点文件加载模型状态字典
    state_dict = torch.load(ckp, map_location=args.device)
    
    # 将状态字典加载到模型中，strict=False允许部分匹配
    model.load_state_dict(state_dict, strict=False)
    
    # 记录模型参数量信息
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    
    # 将模型移动到指定设备
    model = model.to(args.device)
    
    return model, tokenizer



def init_distributed_mode():
    """
    初始化分布式训练模式
    设置分布式训练环境，包括进程组、设备等
    """
    # 如果不是分布式训练则直接返回
    if not ddp: return
    
    # 声明全局变量
    global ddp_local_rank, DEVICE

    # 初始化分布式进程组，使用NCCL后端（适用于GPU）
    dist.init_process_group(backend="nccl")
    
    # 从环境变量获取分布式训练相关参数
    ddp_rank = int(os.environ["RANK"])          # 当前进程的全局rank
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 当前节点上的local rank
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    
    # 设置当前进程使用的设备
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    """
    主函数：模型蒸馏训练入口点
    负责参数解析、环境初始化、数据加载、模型训练等全流程
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="MiniMind Distill Reasoning")
    
    # 添加各种训练参数
    parser.add_argument("--out_dir", type=str, default="../out", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用Weights & Biases记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="W&B项目名称")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载器的工作进程数")
    parser.add_argument("--ddp", action="store_true", help="是否使用分布式数据并行")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热步数")
    parser.add_argument("--log_interval", type=int, default=1, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=50, help="模型保存间隔")
    parser.add_argument('--local_rank', type=int, default=-1, help="本地rank（用于分布式训练）")
    parser.add_argument('--hidden_size', default=512, type=int, help="模型隐藏层大小")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=False, type=bool, help="是否使用MoE（Mixture of Experts）")
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl", help="训练数据路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 创建模型配置对象
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                         use_moe=args.use_moe)
    
    # 设置模型保存目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 计算每个迭代的token数量
    tokens_per_iter = args.batch_size * args.max_seq_len
    
    # 确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置Weights & Biases运行名称
    args.wandb_run_name = f"MiniMind-Distill-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 设置自动混合精度训练上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    # 检查是否为分布式训练
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    
    # 设置随机种子以确保结果可重现
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    # 如果是分布式训练，初始化分布式环境
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    # 如果使用Weights & Biases且是主进程，则初始化wandb
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和tokenizer
    model, tokenizer = init_model(lm_config)

    # 创建训练数据集和数据加载器
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,      # 锁页内存加速数据传输
        drop_last=False,   # 不丢弃最后一个不完整的batch
        shuffle=False,     # 不打乱数据顺序（分布式训练中由sampler负责）
        num_workers=args.num_workers,  # 数据加载的工作进程数
        sampler=train_sampler  # 分布式采样器
    )

    # 创建梯度缩放器（用于混合精度训练）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 如果是分布式训练，包装模型
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}  # 忽略位置编码参数的同步
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    
    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
