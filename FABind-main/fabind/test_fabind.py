# 导入必要的库
import numpy as np  # 用于数值计算
import os  # 用于文件和目录操作
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch  # PyTorch库，用于深度学习
from data import get_data  # 数据加载模块
from torch_geometric.loader import DataLoader  # PyTorch Geometric的数据加载器
from utils.metrics import *  # 导入评估指标相关工具
from utils.utils import *  # 导入通用工具函数
from datetime import datetime  # 用于时间记录
from utils.logging_utils import Logger  # 日志记录工具
import sys  # 用于命令行参数解析
import argparse  # 用于命令行参数解析
import random  # 用于随机数生成
from accelerate import Accelerator  # 用于分布式训练加速
from accelerate import DistributedDataParallelKwargs  # 分布式数据并行参数
from accelerate.utils import set_seed  # 设置随机种子
import shlex  # 用于解析命令行字符串
device = "cuda:1" if torch.cuda.is_available() else "cpu"
# 定义命令行参数解析器
parser = argparse.ArgumentParser(description='FABind model testing.')

# 添加命令行参数
parser.add_argument("-m", "--mode", type=int, default=0, help="指定模型模式。")

parser.add_argument('--seed', type=int, default=600, help="设置随机种子。")
parser.add_argument("--gs-tau", type=float, default=1, help="Gumbel Softmax的温度参数。")
parser.add_argument("--gs-hard", action='store_true', default=False, help="Gumbel Softmax的硬模式。")
parser.add_argument("--batch_size", type=int, default=1, help="测试的批次大小。")
parser.add_argument("--restart", type=str, default=None, help="从头开始训练时加载的模型路径。")
parser.add_argument("--reload", type=str, default=None, help="继续训练时加载的模型路径。")
parser.add_argument("--addNoise", type=str, default=None, help="在训练样本中添加噪声以改变蛋白质口袋的中心位置。")

# 定义互斥参数组，用于交互掩码的选择
pair_interaction_mask = parser.add_mutually_exclusive_group()
pair_interaction_mask.add_argument("--use_y_mask", action='store_true', default=False, help="根据真实的配体-蛋白质交互掩码计算损失。")
pair_interaction_mask.add_argument("--use_equivalent_native_y_mask", action='store_true', default=False, help="根据等效的配体-蛋白质交互掩码计算损失。")

parser.add_argument("--use_affinity_mask", type=int, default=0, help="根据亲和力掩码计算损失。")
parser.add_argument("--affinity_loss_mode", type=int, default=1, help="定义亲和力损失函数的模式。")
parser.add_argument("--pred_dis", type=int, default=1, help="预测距离图或接触图。")
parser.add_argument("--posweight", type=int, default=8, help="接触图损失中的正样本权重。")
parser.add_argument("--relative_k", type=float, default=0.01, help="调整亲和力损失相对于交互损失的权重。")
parser.add_argument("-r", "--relative_k_mode", type=int, default=0, help="定义relative_k随训练轮数的变化模式。")
parser.add_argument("--resultFolder", type=str, default="/data3/Stone/FABind-main/FABind-main/FABind/reproduction_result", help="保存结果的文件夹路径。")
parser.add_argument("--label", type=str, default="", help="记录的附加信息标签。")

parser.add_argument("--use-whole-protein", action='store_true', default=False, help="是否使用整个蛋白质。")

parser.add_argument("--data-path", type=str, default="/data3/Stone/FABind-main/FABind-main/FABind/pdbbind2020", help="数据路径。")
parser.add_argument("-d", "--data", type=str, default="0", help="指定数据模式。0表示重新对接，1表示自对接。")

parser.add_argument("--exp-name", type=str, default="reproduction_experiment", help="实验名称。")
parser.add_argument("--tqdm-interval", type=float, default=0.1, help="进度条更新间隔。")
parser.add_argument("--lr", type=float, default=0.0001, help="学习率。")
parser.add_argument("--pocket-coord-huber-delta", type=float, default=3.0, help="Huber损失函数的delta参数。")
parser.add_argument("--coord-loss-function", type=str, default='SmoothL1', choices=['MSE', 'SmoothL1'], help="坐标损失函数类型。")
parser.add_argument("--coord-loss-weight", type=float, default=1.0, help="坐标损失的权重。")
parser.add_argument("--pair-distance-loss-weight", type=float, default=1.0, help="配体-蛋白质距离损失的权重。")
parser.add_argument("--pair-distance-distill-loss-weight", type=float, default=1.0, help="配体-蛋白质蒸馏损失的权重。")
parser.add_argument("--pocket-cls-loss-weight", type=float, default=1.0, help="蛋白质口袋分类损失的权重。")
parser.add_argument("--pocket-distance-loss-weight", type=float, default=0.05, help="蛋白质口袋距离损失的权重。")
parser.add_argument("--pocket-cls-loss-func", type=str, default='bce', choices=['bce', 'dice'], help="蛋白质口袋分类损失函数类型。")


parser.add_argument('--trig-layers', type=int, default=1, help="三角层的数量。")
parser.add_argument('--distmap-pred', type=str, default='mlp', choices=['mlp', 'trig'], help="距离图预测模型类型。")
parser.add_argument('--mean-layers', type=int, default=3, help="均值层的数量。")
parser.add_argument('--n-iter', type=int, default=5, help="迭代次数。")
parser.add_argument('--inter-cutoff', type=float, default=10.0, help="配体-蛋白质交互的距离阈值。")
parser.add_argument('--intra-cutoff', type=float, default=8.0, help="配体内部交互的距离阈值。")
parser.add_argument('--refine', type=str, default='refine_coord', choices=['stack', 'refine_coord'], help="坐标优化模式。")
parser.add_argument('--coordinate-scale', type=float, default=5.0, help="坐标缩放因子。")
parser.add_argument('--geometry-reg-step-size', type=float, default=0.001, help="几何正则化的步长。")
parser.add_argument('--lr-scheduler', type=str, default="constant", choices=['constant', 'poly_decay', 'cosine_decay', 'cosine_decay_restart', 'exp_decay'], help="学习率调度器类型。")
parser.add_argument('--add-attn-pair-bias', action='store_true', default=False, help="是否添加注意力对偏置。")
parser.add_argument('--explicit-pair-embed', action='store_true', default=False, help="是否显式嵌入配体-蛋白质对。")
parser.add_argument('--opm', action='store_true', default=False, help="是否使用优化模式。")
parser.add_argument('--add-cross-attn-layer', action='store_true', default=False, help="是否添加交叉注意力层。")
parser.add_argument('--rm-layernorm', action='store_true', default=False, help="是否移除层归一化。")
parser.add_argument('--keep-trig-attn', action='store_true', default=False, help="是否保留三角注意力。")
parser.add_argument('--pocket-radius', type=float, default=20.0, help="蛋白质口袋的半径。")
parser.add_argument('--rm-LAS-constrained-optim', action='store_true', default=False, help="是否移除LAS约束优化。")
parser.add_argument('--rm-F-norm', action='store_true', default=False, help="是否移除F范数。")
parser.add_argument('--norm-type', type=str, default="all_sample", choices=['per_sample', '4_sample', 'all_sample'], help="归一化类型。")
parser.add_argument('--noise-for-predicted-pocket', type=float, default=5.0, help="对预测的蛋白质口袋添加噪声。")
parser.add_argument('--test-random-rotation', action='store_true', default=False, help="是否在测试时随机旋转。")
parser.add_argument('--random-n-iter', action='store_true', default=False, help="是否随机迭代次数。")
parser.add_argument('--clip-grad', action='store_true', default=False, help="是否裁剪梯度。")
parser.add_argument("--sample-n", type=int, default=0, help="每轮测试的样本数量。")
parser.add_argument('--fix-pocket', action='store_true', default=False, help="是否固定蛋白质口袋。")
parser.add_argument('--pocket-idx-no-noise', action='store_true', default=False, help="是否对蛋白质口袋索引不添加噪声。")
parser.add_argument('--ablation-no-attention', action='store_true', default=False, help="是否进行无注意力的消融实验。")
parser.add_argument('--ablation-no-attention-with-cross-attn', action='store_true', default=False, help="是否进行无注意力但有交叉注意力的消融实验。")
parser.add_argument('--redocking', action='store_true', default=False, help="是否进行重新对接。")
parser.add_argument('--redocking-no-rotate', action='store_true', default=False, help="是否进行重新对接但不旋转。")
parser.add_argument("--pocket-pred-layers", type=int, default=1, help="蛋白质口袋预测模型的层数。")
parser.add_argument('--pocket-pred-n-iter', type=int, default=1, help="蛋白质口袋预测模型的迭代次数。")
parser.add_argument('--use-esm2-feat', action='store_true', default=False, help="是否使用ESM2特征。")
parser.add_argument("--center-dist-threshold", type=float, default=8.0, help="中心距离阈值。")
parser.add_argument("--mixed-precision", type=str, default='no', choices=['no', 'fp16'], help="是否使用混合精度训练。")
parser.add_argument('--disable-tqdm', action='store_true', default=False, help="是否禁用进度条。")
parser.add_argument('--log-interval', type=int, default=100, help="日志记录间隔。")
parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'], help="优化器类型。")
parser.add_argument("--warmup-epochs", type=int, default=15, help="预热阶段的训练轮数。")
parser.add_argument("--total-epochs", type=int, default=400, help="总训练轮数。")
parser.add_argument('--disable-validate', action='store_true', default=False, help="是否禁用验证阶段。")
parser.add_argument('--disable-tensorboard', action='store_true', default=False, help="是否禁用TensorBoard。")
parser.add_argument("--hidden-size", type=int, default=256, help="隐藏层大小。")
parser.add_argument("--weight-decay", type=float, default=0.0, help="权重衰减系数。")
parser.add_argument("--stage-prob", type=float, default=0.5, help="阶段概率。")
parser.add_argument("--pocket-pred-hidden-size", type=int, default=128, help="蛋白质口袋预测模型的隐藏层大小。")
parser.add_argument("--local-eval", action='store_true', default=False, help="是否启用本地评估模式。")
parser.add_argument("--ckpt", type=str, default='/data3/Stone/FABind-main/FABind-main/FABind/ckpt/best_model_download.bin', help="模型检查点路径。")
parser.add_argument("--train-ligand-torsion-noise", action='store_true', default=False, help="在训练时对配体添加扭转噪声。")
parser.add_argument("--train-pred-pocket-noise", type=float, default=0.0, help="在训练时对预测的蛋白质口袋添加噪声。")
parser.add_argument("--esm2-concat-raw", action='store_true', default=False, help="是否连接原始ESM2特征。")

parser.add_argument("--compound-coords-init-mode", type=str, default="pocket_center_rdkit", choices=['pocket_center_rdkit', 'pocket_center', 'compound_center', 'perturb_3A', 'perturb_4A', 'perturb_5A', 'random'], help="化合物初始坐标模式。")
# 定义化合物初始坐标模式的选项：
# pocket_center_rdkit: 使用 RDKit 生成化合物的三维构象，然后将其几何中心与蛋白质口袋的几何中心对齐。
# pocket_center: 直接使用化合物现有的坐标，将其几何中心与蛋白质口袋的几何中心对齐。
# compound_center: 使用化合物中心作为化合物初始坐标。
# perturb_3A: 在化合物中心添加3埃的扰动。
# perturb_4A: 在化合物中心添加4埃的扰动。
# perturb_5A: 在化合物中心添加5埃的扰动。
# random: 随机生成化合物初始坐标。






args_new = parser.parse_args()

# 解析命令行参数并设置默认值
command = "main_fabind.py -d 0 -m 5 --batch_size 1 --label baseline --addNoise 5 --tqdm-interval 60  --distmap-pred mlp --n-iter 8 --mean-layers 4 --refine refine_coord --coordinate-scale 5 --geometry-reg-step-size 0.001 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --noise-for-predicted-pocket 0.0 --clip-grad --random-n-iter --pocket-idx-no-noise --seed 128 --use-esm2-feat --pocket-pred-layers 1 --pocket-pred-n-iter 1 --center-dist-threshold 4 --pocket-cls-loss-func bce --mixed-precision no --disable-tqdm --disable-validate --log-interval 50 --optim adamw --norm-type per_sample --weight-decay 0.01 --hidden-size 512 --pocket-pred-hidden-size 128 --stage-prob 0.25"
command = shlex.split(command) 

# 使用shlex解析命令行字符串为参数列表

args = parser.parse_args(command[1:])  # 解析命令行参数
args.local_eval = args_new.local_eval  # 设置本地评估模式
args.ckpt = args_new.ckpt  # 设置模型检查点路径
args.data_path = args_new.data_path  # 设置数据路径
args.resultFolder = args_new.resultFolder  # 设置结果保存文件夹路径
args.seed = args_new.seed  # 设置随机种子
args.exp_name = args_new.exp_name  # 设置实验名称
args.batch_size = args_new.batch_size  # 设置批次大小
args.tqdm_interval = 0.1  # 设置进度条更新间隔
args.disable_tqdm = False  # 启用进度条

# 设置随机种子以确保结果可复现
set_seed(args.seed)

# 配置分布式数据并行参数
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # 配置参数以允许未使用的变量
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)  # 初始化加速器

# 设置结果保存路径
pre = f"{args.resultFolder}/{args.exp_name}"  # 构造结果保存路径

# 等待所有进程同步
accelerator.wait_for_everyone()

# 创建结果保存目录
os.makedirs(pre, exist_ok=True)  # 如果目录不存在则创建

# 初始化日志记录器
logger = Logger(accelerator=accelerator, log_path=f'{pre}/test.log')  # 初始化日志记录器

# 记录命令行参数
logger.log_message(f"{' '.join(sys.argv)}")  # 将命令行参数记录到日志

# 设置多进程共享策略为文件系统，解决共享内存问题
torch.multiprocessing.set_sharing_strategy('file_system')  # 设置共享策略

# 根据命令行参数设置化合物初始坐标模式
if args.redocking:
    args.compound_coords_init_mode = "redocking"  # 如果重新对接，则设置模式为"redocking"
elif args.redocking_no_rotate:
    args.redocking = True
    args.compound_coords_init_mode = "redocking_no_rotate"  # 如果重新对接且不旋转，则设置模式为"redocking_no_rotate"

# 加载训练、验证和测试数据集
train, test, test = get_data(
    args, logger, 
    addNoise=args.addNoise,  # 是否添加噪声
    use_whole_protein=args.use_whole_protein,  # 是否使用整个蛋白质
    compound_coords_init_mode=args.compound_coords_init_mode,  # 化合物初始坐标模式
    pre=args.data_path  # 数据路径
)
# logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")  # 记录数据集大小

# 设置数据加载器的工作线程数
num_workers = 10  # 设置线程数为10

# 创建测试数据加载器
test_loader = DataLoader(
    test, batch_size=1, 
    follow_batch=['x', 'compound_pair'],  # 指定需要跟踪的批次数据
    shuffle=False, pin_memory=False, num_workers=num_workers  # 禁用数据打乱，禁用内存锁定
)

# 加载未见过的测试数据
# test_unseen_pdb_list = [line.strip() for line in open('/data3/Stone/FABind-main/FABind-main/FABind/split_pdb_id/unseen_test_index')]  # 从文件中读取未见过的测试数据索引
# test_unseen_index = test.data.query("(group =='test') and (pdb in @test_unseen_pdb_list)").index.values  # 查询未见过的测试数据索引
# test_unseen_index_for_select = np.array([np.where(test._indices == i) for i in test_unseen_index]).reshape(-1)  # 获取索引位置
# test_unseen = test.index_select(test_unseen_index_for_select)  # 根据索引选择数据
# test_unseen_loader = DataLoader(
#     test_unseen, batch_size=args.batch_size, 
#     follow_batch=['x', 'compound_pair'], 
#     shuffle=False, pin_memory=False, num_workers=10  # 创建未见过的测试数据加载器
# )

# 导入模型并设置设备为 GPU
from models.model import *  # 导入模型模块
device = 'cuda:1'  # 设置设备为GPU
model = get_model(args, logger, device)  # 根据参数初始化模型

# 准备模型
model = accelerator.prepare(model)  # 使用加速器准备模型

# 加载模型检查点
model.load_state_dict(torch.load(args.ckpt))  # 加载模型权重

# 设置损失函数
if args.pred_dis:
    criterion = nn.MSELoss()  # 如果预测距离图，则使用均方误差损失
    pred_dis = True
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.posweight))  # 否则使用二分类交叉熵损失

if args.coord_loss_function == 'MSE':
    com_coord_criterion = nn.MSELoss()  # 如果坐标损失函数为MSE，则使用均方误差
elif args.coord_loss_function == 'SmoothL1':
    com_coord_criterion = nn.SmoothL1Loss()  # 如果坐标损失函数为SmoothL1，则使用平滑L1损失

if args.pocket_cls_loss_func == 'bce':
    pocket_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')  # 如果口袋分类损失函数为bce，则使用二分类交叉熵损失

pocket_coord_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)  # 使用Huber损失作为蛋白质口袋坐标损失

model.eval()  # 设置模型为评估模式

logger.log_message(f"Begin test")  # 记录测试开始信息
if accelerator.is_main_process:  # 如果是主进程
    metrics = evaluate_mean_pocket_cls_coord_multi_task(
        accelerator, args, test_loader, accelerator.unwrap_model(model), 
        com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, 
        args.relative_k, accelerator.device, pred_dis=pred_dis, use_y_mask=False, stage=2
    )  # 评估模型性能
    logger.log_stats(metrics, 0, args, prefix="Test_all")  # 记录所有测试数据的评估结果

    # metrics = evaluate_mean_pocket_cls_coord_multi_task(
    #     accelerator, args, test_unseen_loader, accelerator.unwrap_model(model), 
    #     com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, 
    #     args.relative_k, accelerator.device, pred_dis=pred_dis, use_y_mask=False, stage=2
    # )  # 评估未见过的测试数据性能
    # logger.log_stats(metrics, 0, args, prefix="Test_unseen")  # 记录未见过测试数据的评估结果
accelerator.wait_for_everyone()  # 等待所有进程完成