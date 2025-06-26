# filepath: /data3/Stone/FABind-main/FABind-main/FABind/fabind/main_fabind.py

# 导入必要的库
import numpy as np  # 用于数值计算
import os  # 用于文件和目录操作
from tqdm.auto import tqdm  # 用于显示进度条
import torch  # PyTorch库，用于深度学习
from data import get_data  # 数据加载模块
from torch_geometric.loader import DataLoader  # PyTorch Geometric的数据加载器
from utils.metrics import *  # 导入评估指标相关工具
from utils.utils import *  # 导入通用工具函数
from datetime import datetime  # 用于时间记录
from utils.logging_utils import Logger  # 日志记录工具
import sys  # 用于命令行参数解析
import argparse  # 用于命令行参数解析
from torch.utils.data import RandomSampler  # 用于随机采样
import random  # 用于随机数生成
from torch_scatter import scatter_mean  # PyTorch Geometric的工具函数
from utils.metrics_to_tsb import metrics_runtime_no_prefix  # 将指标记录到TensorBoard
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志记录工具
from accelerate import Accelerator  # 用于分布式训练加速
from accelerate import DistributedDataParallelKwargs  # 分布式数据并行参数
from accelerate.utils import set_seed  # 设置随机种子

# 设置随机种子以确保结果可复现
def Seed_everything(seed=42):
    """
    设置随机种子以确保结果可复现。
    参数:
        seed (int): 随机种子值，默认为42。
    """
    random.seed(seed)  # Python随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子
    np.random.seed(seed)  # NumPy随机数种子
    torch.manual_seed(seed)  # PyTorch随机数种子
    torch.cuda.manual_seed(seed)  # CUDA随机数种子
    torch.backends.cudnn.deterministic = True  # 确保CuDNN的确定性

# 定义命令行参数解析器
parser = argparse.ArgumentParser(description='FABind model training.')

# 添加模型相关参数
parser.add_argument("-m", "--mode", type=int, default=0,
                    help="指定模型模式。")
parser.add_argument("-d", "--data", type=str, default="0",
                    help="指定数据模式。0表示重新对接，1表示自对接。")
parser.add_argument('--seed', type=int, default=42,
                    help="设置随机种子。")
parser.add_argument("--gs-tau", type=float, default=1,
                    help="Gumbel Softmax的温度参数。")
parser.add_argument("--gs-hard", action='store_true', default=False,
                    help="Gumbel Softmax的硬模式。")
parser.add_argument("--batch_size", type=int, default=8,
                    help="训练的批次大小。")

parser.add_argument("--restart", type=str, default=None,
                    help="从头开始训练时加载的模型路径。")
parser.add_argument("--reload", type=str, default=None,
                    help="继续训练时加载的模型路径。")
parser.add_argument("--addNoise", type=str, default=None,
                    help="在训练样本中添加噪声以改变蛋白质口袋的中心位置。")

pair_interaction_mask = parser.add_mutually_exclusive_group()
# use_equivalent_native_y_mask is probably a better choice.
pair_interaction_mask.add_argument("--use_y_mask", action='store_true', default=False,
                    help="根据真实的配体-蛋白质交互掩码计算损失。")
pair_interaction_mask.add_argument("--use_equivalent_native_y_mask", action='store_true', default=False,
                    help="根据等效的配体-蛋白质交互掩码计算损失。")

parser.add_argument("--use_affinity_mask", type=int, default=0,
                    help="根据亲和力掩码计算损失。")
parser.add_argument("--affinity_loss_mode", type=int, default=1,
                    help="定义亲和力损失函数的模式。")

parser.add_argument("--pred_dis", type=int, default=1,
                    help="预测距离图或接触图。")
parser.add_argument("--posweight", type=int, default=8,
                    help="接触图损失中的正样本权重。")

parser.add_argument("--relative_k", type=float, default=0.01,
                    help="调整亲和力损失相对于交互损失的权重。")
parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
                    help="定义relative_k随训练轮数的变化模式。")

parser.add_argument("--resultFolder", type=str, default="./result",
                    help="保存结果的文件夹路径。")
parser.add_argument("--label", type=str, default="",
                    help="记录的附加信息标签。")

parser.add_argument("--use-whole-protein", action='store_true', default=False,
                    help="是否使用整个蛋白质。")

parser.add_argument("--data-path", type=str, default="/PDBbind_data/pdbbind2020",
                    help="数据路径。")
                    
parser.add_argument("--exp-name", type=str, default="",
                    help="实验名称。")

parser.add_argument("--tqdm-interval", type=float, default=0.1,
                    help="进度条更新间隔。")

parser.add_argument("--lr", type=float, default=0.0001,
                    help="学习率。")

parser.add_argument("--pocket-coord-huber-delta", type=float, default=3.0,
                    help="Huber损失函数的delta参数。")

parser.add_argument("--coord-loss-function", type=str, default='SmoothL1', choices=['MSE', 'SmoothL1'],
                    help="坐标损失函数类型。")

parser.add_argument("--coord-loss-weight", type=float, default=1.0,
                    help="坐标损失的权重。")
parser.add_argument("--pair-distance-loss-weight", type=float, default=1.0,
                    help="配体-蛋白质距离损失的权重。")
parser.add_argument("--pair-distance-distill-loss-weight", type=float, default=1.0,
                    help="配体-蛋白质蒸馏损失的权重。")
parser.add_argument("--pocket-cls-loss-weight", type=float, default=1.0,
                    help="蛋白质口袋分类损失的权重。")
parser.add_argument("--pocket-distance-loss-weight", type=float, default=0.05,
                    help="蛋白质口袋距离损失的权重。")
parser.add_argument("--pocket-cls-loss-func", type=str, default='bce',
                    help="蛋白质口袋分类损失函数类型。")

parser.add_argument("--compound-coords-init-mode", type=str, default="pocket_center_rdkit",
                    choices=['pocket_center_rdkit', 'pocket_center', 'compound_center', 'perturb_3A', 'perturb_4A', 'perturb_5A', 'random'],
                    help="化合物初始坐标模式。")

parser.add_argument('--trig-layers', type=int, default=1,
                    help="三角层的数量。")

parser.add_argument('--distmap-pred', type=str, default='mlp',
                    choices=['mlp', 'trig'],
                    help="距离图预测模型类型。")

parser.add_argument('--mean-layers', type=int, default=3,
                    help="均值层的数量。")

parser.add_argument('--n-iter', type=int, default=5,
                    help="迭代次数。")

parser.add_argument('--inter-cutoff', type=float, default=10.0,
                    help="配体-蛋白质交互的距离阈值。")

parser.add_argument('--intra-cutoff', type=float, default=8.0,
                    help="配体内部交互的距离阈值。")

parser.add_argument('--refine', type=str, default='refine_coord',
                    choices=['stack', 'refine_coord'],
                    help="坐标优化模式。")

parser.add_argument('--coordinate-scale', type=float, default=5.0,
                    help="坐标缩放因子。")

parser.add_argument('--geometry-reg-step-size', type=float, default=0.001,
                    help="几何正则化的步长。")

parser.add_argument('--lr-scheduler', type=str, default="constant", choices=['constant', 'poly_decay', 'cosine_decay', 'cosine_decay_restart', 'exp_decay'],
                    help="学习率调度器类型。")

parser.add_argument('--add-attn-pair-bias', action='store_true', default=False,
                    help="是否添加注意力对偏置。")

parser.add_argument('--explicit-pair-embed', action='store_true', default=False,
                    help="是否显式嵌入配体-蛋白质对。")

parser.add_argument('--opm', action='store_true', default=False,
                    help="是否使用优化模式。")

parser.add_argument('--add-cross-attn-layer', action='store_true', default=False,
                    help="是否添加交叉注意力层。")

parser.add_argument('--rm-layernorm', action='store_true', default=False,
                    help="是否移除层归一化。")

parser.add_argument('--keep-trig-attn', action='store_true', default=False,
                    help="是否保留三角注意力。")

parser.add_argument('--pocket-radius', type=float, default=20.0,
                    help="蛋白质口袋的半径。")

parser.add_argument('--rm-LAS-constrained-optim', action='store_true', default=False,
                    help="是否移除LAS约束优化。")

parser.add_argument('--rm-F-norm', action='store_true', default=False,
                    help="是否移除F范数。")

parser.add_argument('--norm-type', type=str, default="per_sample", choices=['per_sample', '4_sample', 'all_sample'],
                    help="归一化类型。")

parser.add_argument('--noise-for-predicted-pocket', type=float, default=5.0,
                    help="对预测的蛋白质口袋添加噪声。")

parser.add_argument('--test-random-rotation', action='store_true', default=False,
                    help="是否在测试时随机旋转。")

parser.add_argument('--random-n-iter', action='store_true', default=False,
                    help="是否随机迭代次数。")

parser.add_argument('--clip-grad', action='store_true', default=False,
                    help="是否裁剪梯度。")

parser.add_argument("--sample-n", type=int, default=0, help="每轮训练的样本数量。")

parser.add_argument('--fix-pocket', action='store_true', default=False,
                    help="是否固定蛋白质口袋。")

parser.add_argument('--pocket-idx-no-noise', action='store_true', default=False,
                    help="是否对蛋白质口袋索引不添加噪声。")

parser.add_argument('--ablation-no-attention', action='store_true', default=False,
                    help="是否进行无注意力的消融实验。")

parser.add_argument('--ablation-no-attention-with-cross-attn', action='store_true', default=False,
                    help="是否进行无注意力但有交叉注意力的消融实验。")

parser.add_argument('--redocking', action='store_true', default=False,
                    help="是否进行重新对接。")

parser.add_argument('--redocking-no-rotate', action='store_true', default=False,
                    help="是否进行重新对接但不旋转。")

parser.add_argument("--pocket-pred-layers", type=int, default=1, help="蛋白质口袋预测模型的层数。")
parser.add_argument('--pocket-pred-n-iter', type=int, default=1, help="蛋白质口袋预测模型的迭代次数。")

parser.add_argument('--use-esm2-feat', action='store_true', default=False,
                    help="是否使用ESM2特征。")

parser.add_argument("--center-dist-threshold", type=float, default=8.0,
                    help="中心距离阈值。")

parser.add_argument("--mixed-precision", type=str, default='no', choices=['no', 'fp16'],
                    help="是否使用混合精度训练。")

parser.add_argument('--disable-tqdm', action='store_true', default=False,
                    help="是否禁用进度条。")

parser.add_argument('--log-interval', type=int, default=100,
                    help="日志记录间隔。")

parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'],
                    help="优化器类型。")

parser.add_argument("--warmup-epochs", type=int, default=15,
                    help="预热阶段的训练轮数。")

parser.add_argument("--total-epochs", type=int, default=400,
                    help="总训练轮数。")

parser.add_argument('--disable-validate', action='store_true', default=False,
                    help="是否禁用验证阶段。")

parser.add_argument('--disable-tensorboard', action='store_true', default=False,
                    help="是否禁用TensorBoard。")

parser.add_argument("--hidden-size", type=int, default=256,
                    help="隐藏层大小。")

parser.add_argument("--weight-decay", type=float, default=0.0,
                    help="权重衰减系数。")

parser.add_argument("--stage-prob", type=float, default=0.5,
                    help="阶段概率。")

parser.add_argument("--pocket-pred-hidden-size", type=int, default=128,
                    help="蛋白质口袋预测模型的隐藏层大小。")

parser.add_argument("--local-eval", action='store_true', default=False,
                    help="是否启用本地评估模式。")

parser.add_argument("--train-ligand-torsion-noise", action='store_true', default=False,
                    help="在训练时对配体添加扭转噪声。")

parser.add_argument("--train-pred-pocket-noise", type=float, default=0.0,
                    help="在训练时对预测的蛋白质口袋添加噪声。")

parser.add_argument('--esm2-concat-raw', action='store_true', default=False,
                    help="是否连接原始ESM2特征。")

args = parser.parse_args()

# 配置分布式数据并行参数
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)
set_seed(args.seed)  # 设置种子以确保结果可复现
pre = f"{args.resultFolder}/{args.exp_name}"

# 如果是主进程，创建必要的目录
if accelerator.is_main_process:
    os.system(f"mkdir -p {pre}/models")

    # 如果未禁用TensorBoard，则创建日志目录
    if not args.disable_tensorboard:
        tsb_runtime_dir = f"{pre}/tsb_runtime"
        os.system(f"mkdir -p {tsb_runtime_dir}")
        train_writer = SummaryWriter(log_dir=f'{tsb_runtime_dir}/train')  # 训练日志
        valid_writer = SummaryWriter(log_dir=f'{tsb_runtime_dir}/valid')  # 验证日志
        test_writer = SummaryWriter(log_dir=f'{tsb_runtime_dir}/test')  # 测试日志
        test_writer_use_predicted_pocket = SummaryWriter(log_dir=f'{tsb_runtime_dir}/test_use_predicted_pocket')

accelerator.wait_for_everyone()

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

# 设置日志记录器
logger = Logger(accelerator=accelerator, log_path=f'{pre}/{timestamp}.log')

# 记录命令行参数
logger.log_message(f"{' '.join(sys.argv)}")

# 设置多进程共享策略为文件系统，解决共享内存问题
torch.multiprocessing.set_sharing_strategy('file_system')

# 根据命令行参数设置化合物初始坐标模式
if args.redocking:
    args.compound_coords_init_mode = "redocking"
elif args.redocking_no_rotate:
    args.redocking = True
    args.compound_coords_init_mode = "redocking_no_rotate"

# 加载训练、验证和测试数据集
train, valid, test = get_data(
    args, logger, 
    addNoise=args.addNoise, 
    use_whole_protein=args.use_whole_protein, 
    compound_coords_init_mode=args.compound_coords_init_mode, 
    pre=args.data_path
)
logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

# 设置数据加载器的工作线程数
num_workers = 10

# 根据样本数量设置数据加载器
if args.sample_n > 0:
    sampler = RandomSampler(train, replacement=True, num_samples=args.sample_n)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, 
        follow_batch=['x', 'compound_pair'], 
        sampler=sampler, pin_memory=False, num_workers=num_workers
    )
else:
    train_loader = DataLoader(
        train, batch_size=args.batch_size, 
        follow_batch=['x', 'compound_pair'], 
        shuffle=True, pin_memory=False, num_workers=num_workers
    )

# 设置验证和测试数据加载器
valid_loader = DataLoader(
    valid, batch_size=args.batch_size, 
    follow_batch=['x', 'compound_pair'], 
    shuffle=False, pin_memory=False, num_workers=num_workers
)
test_loader = DataLoader(
    test, batch_size=args.batch_size, 
    follow_batch=['x', 'compound_pair'], 
    shuffle=False, pin_memory=False, num_workers=num_workers
)

# 导入模型并设置设备为 GPU
from models.model import *
device = 'cuda'
model = get_model(args, logger, device)

# 根据优化器类型设置优化器
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 设置学习率调度器
last_epoch = -1
steps_per_epoch = len(train_loader)
total_training_steps = args.total_epochs * len(train_loader)

# 预热阶段的学习率调度器
scheduler_warm_up = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.5, end_factor=1, 
    total_iters=args.warmup_epochs * len(train_loader), 
    last_epoch=last_epoch
)

# 主训练阶段的学习率调度器
if args.lr_scheduler == "constant":
    scheduler_post = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, 
        total_iters=(args.total_epochs - args.warmup_epochs) * len(train_loader), 
        last_epoch=last_epoch
    )
elif args.lr_scheduler == "poly_decay":
    scheduler_post = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, 
        total_iters=(args.total_epochs - args.warmup_epochs) * len(train_loader), 
        last_epoch=last_epoch
    )
elif args.lr_scheduler == "exp_decay":
    scheduler_post = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.995, last_epoch=last_epoch
    )
elif args.lr_scheduler == "cosine_decay":
    scheduler_post = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(args.total_epochs - args.warmup_epochs) * len(train_loader), 
        eta_min=1e-5, last_epoch=last_epoch
    )
elif args.lr_scheduler == "cosine_decay_restart":
    scheduler_post = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=0.0001, last_epoch=last_epoch
    )

# 合并预热和主训练阶段的调度器
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[scheduler_warm_up, scheduler_post], 
    milestones=[args.warmup_epochs * len(train_loader)]
)

# 准备模型、优化器、调度器和数据加载器
(
    model,
    optimizer,
    scheduler,
    train_loader,
) = accelerator.prepare(
    model, optimizer, scheduler, train_loader,
)

# 检查是否有已保存的模型状态
output_last_epoch_dir = f"{pre}/models/epoch_last"
if os.path.exists(output_last_epoch_dir) and os.path.exists(os.path.join(output_last_epoch_dir, "pytorch_model.bin")):
    accelerator.load_state(output_last_epoch_dir)
    last_epoch = round(scheduler.state_dict()['last_epoch'] / steps_per_epoch) - 1
    logger.log_message(f'Load model from epoch: {last_epoch}')

# 设置损失函数
if args.pred_dis:
    criterion = nn.MSELoss()
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.posweight))

if args.coord_loss_function == 'MSE':
    com_coord_criterion = nn.MSELoss()
elif args.coord_loss_function == 'SmoothL1':
    com_coord_criterion = nn.SmoothL1Loss()

if args.pocket_cls_loss_func == 'bce':
    pocket_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')

pocket_coord_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)

# 初始化训练指标
best_auroc = 0
best_f1_1 = 0
epoch_not_improving = 0

logger.log_message(f"Total epochs: {args.total_epochs}")
logger.log_message(f"Total training steps: {total_training_steps}")

# 开始训练循环
for epoch in range(last_epoch + 1, args.total_epochs):
    model.train()
    
    # 初始化批次指标
    y_list = []
    y_pred_list = []
    com_coord_list = []
    com_coord_pred_list = []
    rmsd_list = []
    rmsd_2A_list = []
    rmsd_5A_list = []
    centroid_dis_list = []
    centroid_dis_2A_list = []
    centroid_dis_5A_list = []
    pocket_coord_list = []
    pocket_coord_pred_list = []
    pocket_cls_list = []
    pocket_cls_pred_list = []
    pocket_cls_pred_round_list = []
    protein_len_list = []
    count = 0
    skip_count = 0
    batch_loss = 0.0
    batch_by_pred_loss = 0.0
    batch_distill_loss = 0.0
    com_coord_batch_loss = 0.0
    pocket_cls_batch_loss = 0.0
    pocket_coord_batch_loss = 0.0
    keepNode_less_5_count = 0

    # 设置数据迭代器
    if args.disable_tqdm:
        data_iter = train_loader
    else:
        data_iter = tqdm(train_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)

    # 批次训练
    for batch_id, data in enumerate(data_iter, start=1):
        optimizer.zero_grad()

        # 模型前向传播
        com_coord_pred, compound_batch, y_pred, y_pred_by_coord, pocket_cls_pred, pocket_cls, protein_out_mask_whole, p_coords_batched_whole, pred_pocket_center, dis_map, keepNode_less_5 = model(data, train=True)

        # 计算损失
        pocket_cls_loss = args.pocket_cls_loss_weight * pocket_cls_criterion(pocket_cls_pred, pocket_cls.float()) * (protein_out_mask_whole.numel() / protein_out_mask_whole.sum())
        pocket_coord_loss = args.pocket_distance_loss_weight * pocket_coord_criterion(pred_pocket_center, data.coords_center)
        contact_loss = args.pair_distance_loss_weight * criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0])
        contact_by_pred_loss = args.pair_distance_loss_weight * criterion(y_pred_by_coord, dis_map) if len(dis_map) > 0 else torch.tensor([0])
        contact_distill_loss = args.pair_distance_distill_loss_weight * criterion(y_pred_by_coord, y_pred) if len(y_pred) > 0 else torch.tensor([0])
        com_coord_loss = args.coord_loss_weight * com_coord_criterion(com_coord_pred, com_coord) if len(com_coord) > 0 else torch.tensor([0])

        # 总损失
        loss = com_coord_loss + contact_loss + contact_by_pred_loss + contact_distill_loss + pocket_cls_loss + pocket_coord_loss

        # 反向传播
        accelerator.backward(loss)
        if args.clip_grad:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新优化器和调度器
        optimizer.step()
        scheduler.step()

        # 记录批次指标
        batch_loss += len(y_pred) * contact_loss.item()
        batch_by_pred_loss += len(y_pred_by_coord) * contact_by_pred_loss.item()
        batch_distill_loss += len(y_pred_by_coord) * contact_distill_loss.item()
        com_coord_batch_loss += len(com_coord_pred) * com_coord_loss.item()
        pocket_cls_batch_loss += len(pocket_cls_pred) * pocket_cls_loss.item()
        pocket_coord_batch_loss += len(pred_pocket_center) * pocket_coord_loss.item()
        keepNode_less_5_count += keepNode_less_5

        # 记录样本数据
        y_list.append(dis_map.detach())
        y_pred_list.append(y_pred.detach())
        com_coord_list.append(com_coord)
        com_coord_pred_list.append(com_coord_pred.detach())
        rmsd_list.append(rmsd.detach())
        rmsd_2A_list.append((rmsd.detach() < 2).float())
        rmsd_5A_list.append((rmsd.detach() < 5).float())
        centroid_dis_list.append(centroid_dis.detach())
        centroid_dis_2A_list.append((centroid_dis.detach() < 2).float())
        centroid_dis_5A_list.append((centroid_dis.detach() < 5).float())

        # 记录日志
        if batch_id % args.log_interval == 0:
            stats_dict = {
                'step': batch_id,
                'lr': optimizer.param_groups[0]['lr'],
                'contact_loss': contact_loss.item(),
                'contact_by_pred_loss': contact_by_pred_loss.item(),
                'contact_distill_loss': contact_distill_loss.item(),
                'com_coord_loss': com_coord_loss.item(),
                'pocket_cls_loss': pocket_cls_loss.item(),
                'pocket_coord_loss': pocket_coord_loss.item(),
            }
            logger.log_stats(stats_dict, epoch, args, prefix='train')

    # 释放内存
    y, y_pred = None, None
    com_coord, com_coord_pred = None, None
    rmsd, rmsd_2A, rmsd_5A = None, None, None
    centroid_dis, centroid_dis_2A, centroid_dis_5A = None, None, None
    pocket_cls, pocket_cls_pred, pocket_cls_pred_round, pocket_coord_pred, pocket_coord, protein_len = None, None, None, None, None, None

    # 验证和测试
    model.eval()
    logger.log_message(f"Begin validation")
    if accelerator.is_main_process:
        if not args.disable_validate:
            metrics = evaluate_mean_pocket_cls_coord_multi_task(
                accelerator, args, valid_loader, model, 
                com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, 
                args.relative_k, device, pred_dis=pred_dis, use_y_mask=False, stage=1
            )
            logger.log_stats(metrics, epoch, args, prefix="Valid")

    logger.log_message(f"Begin test")
    if accelerator.is_main_process:
        metrics = evaluate_mean_pocket_cls_coord_multi_task(
            accelerator, args, test_loader, accelerator.unwrap_model(model), 
            com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, 
            args.relative_k, accelerator.device, pred_dis=pred_dis, use_y_mask=False, stage=1
        )
        logger.log_stats(metrics, epoch, args, prefix="Test")

        metrics = evaluate_mean_pocket_cls_coord_multi_task(
            accelerator, args, test_loader, accelerator.unwrap_model(model), 
            com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, 
            args.relative_k, accelerator.device, pred_dis=pred_dis, use_y_mask=False, stage=2
        )
        logger.log_stats(metrics, epoch, args, prefix="Test_pp")

        # 保存模型状态
        output_dir = f"{pre}/models/epoch_{epoch}"
        accelerator.save_state(output_dir=output_dir)
        accelerator.save_state(output_dir=output_last_epoch_dir)

    accelerator.wait_for_everyone()

