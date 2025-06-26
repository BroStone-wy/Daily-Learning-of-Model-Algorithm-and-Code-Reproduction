import numpy as np  # 导入NumPy库，用于数值计算
import os  # 导入os库，用于操作文件和目录

import torch  # 导入PyTorch库，用于深度学习

from torch_geometric.loader import DataLoader  # 从PyTorch Geometric导入DataLoader，用于加载图数据
from datetime import datetime  # 导入datetime模块，用于处理日期和时间
from utils.logging_utils import Logger  # 从utils模块导入Logger类，用于日志记录
import sys  # 导入sys模块，用于操作Python运行时环境
import argparse  # 导入argparse模块，用于解析命令行参数
import random  # 导入random模块，用于生成随机数
from accelerate import Accelerator  # 从accelerate库导入Accelerator类，用于加速训练
from accelerate import DistributedDataParallelKwargs  # 导入DistributedDataParallelKwargs，用于分布式训练配置
from accelerate.utils import set_seed  # 导入set_seed函数，用于设置随机种子
import shlex  # 导入shlex模块，用于解析命令行字符串
import glob  # 导入glob模块，用于文件路径匹配
import time  # 导入time模块，用于计时
import pathlib  # 导入pathlib模块，用于操作路径

from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

from utils.fabind_inference_dataset import InferenceDataset  # 从utils模块导入InferenceDataset类，用于推理数据集
from utils.inference_mol_utils import write_mol  # 从utils模块导入write_mol函数，用于写入分子文件
from utils.post_optim_utils import post_optimize_compound_coords  # 从utils模块导入post_optimize_compound_coords函数，用于后优化分子坐标
import pandas as pd  # 导入Pandas库，用于数据处理

def Seed_everything(seed=42):  # 定义一个函数，用于设置随机种子
    random.seed(seed)  # 设置Python的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置环境变量的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    torch.cuda.manual_seed(seed)  # 设置CUDA的随机种子
    torch.backends.cudnn.deterministic = True  # 设置CuDNN为确定性模式


parser = argparse.ArgumentParser(description='Train your own TankBind model.')  # 创建一个命令行参数解析器

parser.add_argument("-m", "--mode", type=int, default=0,
                    help="mode specify the model to use.")
parser.add_argument("-d", "--data", type=str, default="0",
                    help="data specify the data to use. \
                    0 for re-docking, 1 for self-docking.")
parser.add_argument('--seed', type=int, default=42,
                    help="seed to use.")
parser.add_argument("--gs-tau", type=float, default=1,
                    help="Tau for the temperature-based softmax.")
parser.add_argument("--gs-hard", action='store_true', default=False,
                    help="Hard mode for gumbel softmax.")
parser.add_argument("--batch_size", type=int, default=8,
                    help="batch size.")

parser.add_argument("--restart", type=str, default=None,
                    help="continue the training from the model we saved from scratch.")
parser.add_argument("--reload", type=str, default=None,
                    help="continue the training from the model we saved.")
parser.add_argument("--addNoise", type=str, default=None,
                    help="shift the location of the pocket center in each training sample \
                    such that the protein pocket encloses a slightly different space.")

pair_interaction_mask = parser.add_mutually_exclusive_group()
# use_equivalent_native_y_mask is probably a better choice.
pair_interaction_mask.add_argument("--use_y_mask", action='store_true', default=False,
                    help="mask the pair interaction during pair interaction loss evaluation based on data.real_y_mask. \
                    real_y_mask=True if it's the native pocket that ligand binds to.")
pair_interaction_mask.add_argument("--use_equivalent_native_y_mask", action='store_true', default=False,
                    help="mask the pair interaction during pair interaction loss evaluation based on data.equivalent_native_y_mask. \
                    real_y_mask=True if most of the native interaction between ligand and protein happen inside this pocket.")

parser.add_argument("--use_affinity_mask", type=int, default=0,
                    help="mask affinity in loss evaluation based on data.real_affinity_mask")
parser.add_argument("--affinity_loss_mode", type=int, default=1,
                    help="define which affinity loss function to use.")

parser.add_argument("--pred_dis", type=int, default=1,
                    help="pred distance map or predict contact map.")
parser.add_argument("--posweight", type=int, default=8,
                    help="pos weight in pair contact loss, not useful if args.pred_dis=1")

parser.add_argument("--relative_k", type=float, default=0.01,
                    help="adjust the strength of the affinity loss head relative to the pair interaction loss.")
parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
                    help="define how the relative_k changes over epochs")

parser.add_argument("--resultFolder", type=str, default="./result",
                    help="information you want to keep a record.")
parser.add_argument("--label", type=str, default="",
                    help="information you want to keep a record.")

parser.add_argument("--use-whole-protein", action='store_true', default=False,
                    help="currently not used.")

parser.add_argument("--data-path", type=str, default="",
                    help="Data path.")
                    
parser.add_argument("--exp-name", type=str, default="",
                    help="data path.")

parser.add_argument("--tqdm-interval", type=float, default=0.1,
                    help="tqdm bar update interval")

parser.add_argument("--lr", type=float, default=0.0001)

parser.add_argument("--pocket-coord-huber-delta", type=float, default=3.0)

parser.add_argument("--coord-loss-function", type=str, default='SmoothL1', choices=['MSE', 'SmoothL1'])

parser.add_argument("--coord-loss-weight", type=float, default=1.0)
parser.add_argument("--pair-distance-loss-weight", type=float, default=1.0)
parser.add_argument("--pair-distance-distill-loss-weight", type=float, default=1.0)
parser.add_argument("--pocket-cls-loss-weight", type=float, default=1.0)
parser.add_argument("--pocket-distance-loss-weight", type=float, default=0.05)
parser.add_argument("--pocket-cls-loss-func", type=str, default='bce', choices=['bce', 'dice'])

# parser.add_argument("--warm-mae-thr", type=float, default=5.0)

parser.add_argument("--use-compound-com-cls", action='store_true', default=False,
                    help="only use real pocket to run pocket classification task")

parser.add_argument("--compound-coords-init-mode", type=str, default="pocket_center_rdkit",
                    choices=['pocket_center_rdkit', 'pocket_center', 'compound_center', 'perturb_3A', 'perturb_4A', 'perturb_5A', 'random', 'diffdock'])

parser.add_argument('--trig-layers', type=int, default=1)

parser.add_argument('--distmap-pred', type=str, default='mlp',
                    choices=['mlp', 'trig'])
parser.add_argument('--mean-layers', type=int, default=3)
parser.add_argument('--n-iter', type=int, default=8)
parser.add_argument('--inter-cutoff', type=float, default=10.0)
parser.add_argument('--intra-cutoff', type=float, default=8.0)
parser.add_argument('--refine', type=str, default='refine_coord',
                    choices=['stack', 'refine_coord'])

parser.add_argument('--coordinate-scale', type=float, default=5.0)
parser.add_argument('--geometry-reg-step-size', type=float, default=0.001)
parser.add_argument('--lr-scheduler', type=str, default="constant", choices=['constant', 'poly_decay', 'cosine_decay', 'cosine_decay_restart', 'exp_decay'])

parser.add_argument('--add-attn-pair-bias', action='store_true', default=False)
parser.add_argument('--explicit-pair-embed', action='store_true', default=False)
parser.add_argument('--opm', action='store_true', default=False)

parser.add_argument('--add-cross-attn-layer', action='store_true', default=False)
parser.add_argument('--rm-layernorm', action='store_true', default=False)
parser.add_argument('--keep-trig-attn', action='store_true', default=False)

parser.add_argument('--pocket-radius', type=float, default=20.0)

parser.add_argument('--rm-LAS-constrained-optim', action='store_true', default=False)
parser.add_argument('--rm-F-norm', action='store_true', default=False)
parser.add_argument('--norm-type', type=str, default="per_sample", choices=['per_sample', '4_sample', 'all_sample'])

# parser.add_argument("--only-predicted-pocket-mae-thr", type=float, default=3.0)
parser.add_argument('--noise-for-predicted-pocket', type=float, default=5.0)
parser.add_argument('--test-random-rotation', action='store_true', default=False)

parser.add_argument('--random-n-iter', action='store_true', default=False)
parser.add_argument('--clip-grad', action='store_true', default=False)

# one batch actually contains 20000 samples, not the size of training set
parser.add_argument("--sample-n", type=int, default=0, help="number of samples in one epoch.")

parser.add_argument('--fix-pocket', action='store_true', default=False)
parser.add_argument('--pocket-idx-no-noise', action='store_true', default=False)
parser.add_argument('--ablation-no-attention', action='store_true', default=False)
parser.add_argument('--ablation-no-attention-with-cross-attn', action='store_true', default=False)

parser.add_argument('--redocking', action='store_true', default=False)
parser.add_argument('--redocking-no-rotate', action='store_true', default=False)

parser.add_argument("--pocket-pred-layers", type=int, default=1, help="number of layers for pocket pred model.")
parser.add_argument('--pocket-pred-n-iter', type=int, default=1, help="number of iterations for pocket pred model.")

parser.add_argument('--use-esm2-feat', action='store_true', default=False)
parser.add_argument("--center-dist-threshold", type=float, default=8.0)

parser.add_argument("--mixed-precision", type=str, default='no', choices=['no', 'fp16'])
parser.add_argument('--disable-tqdm', action='store_true', default=False)
parser.add_argument('--log-interval', type=int, default=100)
parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'])
parser.add_argument("--warmup-epochs", type=int, default=15,
                    help="used in combination with relative_k_mode.")
parser.add_argument("--total-epochs", type=int, default=400,
                    help="option to switch training data after certain epochs.")
parser.add_argument('--disable-validate', action='store_true', default=False)
parser.add_argument('--disable-tensorboard', action='store_true', default=False)
parser.add_argument("--hidden-size", type=int, default=256)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--stage-prob", type=float, default=0.5)
parser.add_argument("--pocket-pred-hidden-size", type=int, default=128)

parser.add_argument("--local-eval", action='store_true', default=False)
# parser.add_argument("--eval-dir", type=str, default=None)

parser.add_argument("--train-ligand-torsion-noise", action='store_true', default=False)
parser.add_argument("--train-pred-pocket-noise", type=float, default=0.0)
parser.add_argument("--esm2-concat-raw", action='store_true', default=False)
parser.add_argument("--test-sample-n", type=int, default=1)
parser.add_argument("--return-hidden", action='store_true', default=False)
parser.add_argument("--confidence-task", type=str, default='classification', choices=['classification', 'regression', 'perfect'])
parser.add_argument("--confidence-rmsd-thr", type=float, default=2.0)
parser.add_argument("--confidence-thr", type=float, default=0.5)

parser.add_argument("--post-optim", action='store_true', default=False)
parser.add_argument('--post-optim-mode', type=int, default=0)
parser.add_argument('--post-optim-epoch', type=int, default=1000)
parser.add_argument("--rigid", action='store_true', default=False)

parser.add_argument("--ensemble", action='store_true', default=False)
parser.add_argument("--confidence", action='store_true', default=False)
parser.add_argument("--test-gumbel-soft", action='store_true', default=False)
parser.add_argument("--test-pocket-noise", type=float, default=5)
parser.add_argument("--test-unseen", action='store_true', default=False)

parser.add_argument('--sdf-output-path-post-optim', type=str, default="")
parser.add_argument('--write-mol-to-file', action='store_true', default=False)
parser.add_argument('--sdf-to-mol2', action='store_true', default=False)

parser.add_argument('--index-csv', type=str, default=None)
parser.add_argument('--pdb-file-dir', type=str, default="")
parser.add_argument('--preprocess-dir', type=str, default="")
parser.add_argument("--ckpt", type=str, default='../checkpoints/pytorch_model.bin')

args_new = parser.parse_args()  # 解析命令行参数

command = "main_fabind.py -d 0 -m 5 --batch_size 3 --label baseline --addNoise 5 --tqdm-interval 60 --use-compound-com-cls --distmap-pred mlp --n-iter 8 --mean-layers 4 --refine refine_coord --coordinate-scale 5 --geometry-reg-step-size 0.001 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --noise-for-predicted-pocket 0.0 --clip-grad --random-n-iter --pocket-idx-no-noise --seed 128 --use-esm2-feat --pocket-pred-layers 1 --pocket-pred-n-iter 1 --center-dist-threshold 4 --pocket-cls-loss-func bce --mixed-precision no --disable-tqdm --disable-validate --log-interval 50 --optim adamw --norm-type per_sample --weight-decay 0.01 --hidden-size 512 --pocket-pred-hidden-size 128 --stage-prob 0.25"  # 定义一个命令字符串
command = shlex.split(command)  # 使用shlex解析命令字符串

args = parser.parse_args(command[1:])  # 解析命令行参数
args.local_eval = args_new.local_eval  # 将新参数赋值给args
# args.eval_dir = args_new.eval_dir
args.batch_size = args_new.batch_size
args.ckpt = args_new.ckpt
args.data_path = args_new.data_path
args.resultFolder = args_new.resultFolder
args.seed = args_new.seed
args.exp_name = args_new.exp_name
args.return_hidden = args_new.return_hidden
args.confidence_task = args_new.confidence_task
args.confidence_rmsd_thr = args_new.confidence_rmsd_thr
args.confidence_thr = args_new.confidence_thr
args.test_sample_n = args_new.test_sample_n
args.disable_tqdm = False
args.tqdm_interval = 0.1
args.train_pred_pocket_noise = args_new.train_pred_pocket_noise
args.post_optim = args_new.post_optim
args.post_optim_mode = args_new.post_optim_mode
args.post_optim_epoch = args_new.post_optim_epoch
args.rigid = args_new.rigid
args.ensemble = args_new.ensemble
args.confidence = args_new.confidence
args.test_gumbel_soft = args_new.test_gumbel_soft
args.test_pocket_noise = args_new.test_pocket_noise
args.test_unseen = args_new.test_unseen
args.gs_tau = args_new.gs_tau
args.compound_coords_init_mode = args_new.compound_coords_init_mode
args.sdf_output_path_post_optim = args_new.sdf_output_path_post_optim
args.write_mol_to_file = args_new.write_mol_to_file
args.sdf_to_mol2 = args_new.sdf_to_mol2
args.n_iter = args_new.n_iter
args.redocking = args_new.redocking

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # 配置分布式训练参数
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)  # 初始化加速器

pre = f"{args.resultFolder}/{args.exp_name}"  # 定义结果文件夹路径

os.makedirs(args.sdf_output_path_post_optim, exist_ok=True)  # 创建SDF输出路径
os.makedirs(pre, exist_ok=True)  # 创建结果文件夹
logger = Logger(accelerator=accelerator, log_path=f'{pre}/test.log')  # 初始化日志记录器

logger.log_message(f"{' '.join(sys.argv)}")  # 记录命令行参数

# torch.set_num_threads(16)
# # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy('file_system')  # 设置多进程共享策略为文件系统

# train, valid, test: only native pocket. train_after_warm_up, all_pocket_test include all other pockets(protein center and P2rank result)
if args.redocking:
    args.compound_coords_init_mode = "redocking"
elif args.redocking_no_rotate:
    args.redocking = True
    args.compound_coords_init_mode = "redocking_no_rotate"


def post_optim_mol(args, accelerator, data, com_coord_pred, com_coord_pred_per_sample_list, com_coord_per_sample_list, compound_batch, LAS_tmp, rigid=False):
    post_optim_device='cpu'  # 定义后优化操作的设备为CPU
    for i in range(compound_batch.max().item()+1):  # 遍历每个样本的批次
        i_mask = (compound_batch == i)  # 获取当前样本的掩码
        com_coord_pred_i = com_coord_pred[i_mask]  # 获取当前样本的预测坐标
        com_coord_i = data[i]['compound'].rdkit_coords  # 获取当前样本的真实坐标

        com_coord_pred_center_i = com_coord_pred_i.mean(dim=0).reshape(1, 3)  # 计算预测坐标的中心点
        
        if rigid:  # 如果启用了刚性优化
            predict_coord, loss, rmsd = post_optimize_compound_coords(
                reference_compound_coords=com_coord_i.to(post_optim_device),  # 将真实坐标移动到后优化设备
                predict_compound_coords=com_coord_pred_i.to(post_optim_device),  # 将预测坐标移动到后优化设备
                LAS_edge_index=None,  # 不使用LAS边索引
                mode=args.post_optim_mode,  # 后优化模式
                total_epoch=args.post_optim_epoch,  # 后优化的总迭代次数
            )
            predict_coord.to(accelerator.device)  # 将优化后的坐标移动到加速器设备
            predict_coord = predict_coord - predict_coord.mean(dim=0).reshape(1, 3) + com_coord_pred_center_i  # 调整优化后的坐标中心
            com_coord_pred[i_mask] = predict_coord  # 更新预测坐标
        else:  # 如果未启用刚性优化
            predict_coord, loss, rmsd = post_optimize_compound_coords(
                reference_compound_coords=com_coord_i.to(post_optim_device),  # 将真实坐标移动到后优化设备
                predict_compound_coords=com_coord_pred_i.to(post_optim_device),  # 将预测坐标移动到后优化设备
                LAS_edge_index=LAS_tmp[i].to(post_optim_device),  # 使用LAS边索引进行优化
                mode=args.post_optim_mode,  # 后优化模式
                total_epoch=args.post_optim_epoch,  # 后优化的总迭代次数
            )
            predict_coord = predict_coord.to(accelerator.device)  # 将优化后的坐标移动到加速器设备
            predict_coord = predict_coord - predict_coord.mean(dim=0).reshape(1, 3) + com_coord_pred_center_i  # 调整优化后的坐标中心
            com_coord_pred[i_mask] = predict_coord  # 更新预测坐标
        
        com_coord_pred_per_sample_list.append(com_coord_pred[i_mask])  # 保存每个样本的预测坐标
        com_coord_per_sample_list.append(com_coord_i)  # 保存每个样本的真实坐标
        com_coord_offset_per_sample_list.append(data[i].coord_offset)  # 保存每个样本的坐标偏移量
        
        mol_list.append(data[i].mol)  # 保存分子对象
        uid_list.append(data[i].uid)  # 保存唯一标识符
        smiles_list.append(data[i]['compound'].smiles)  # 保存SMILES字符串
        sdf_name_list.append(data[i].ligand_id + '.sdf')  # 保存SDF文件名

    return  # 返回函数结束


dataset = InferenceDataset(args_new.index_csv, args_new.pdb_file_dir, args_new.preprocess_dir)  # 初始化推理数据集
logger.log_message(f"data point: {len(dataset)}")  # 记录数据点数量
num_workers = 0  # 设置数据加载器的工作线程数
data_loader = DataLoader(dataset, batch_size=args.batch_size, follow_batch=['x'], shuffle=False, pin_memory=False, num_workers=num_workers)  # 初始化数据加载器

device = 'cuda'  # 设置设备为CUDA
from models.model import *  # 导入模型模块
model = get_model(args, logger, device)  # 获取模型实例

model = accelerator.prepare(model)  # 准备模型以进行加速

model.load_state_dict(torch.load(args.ckpt))  # 加载模型权重

set_seed(args.seed)  # 设置随机种子

model.eval()  # 设置模型为评估模式

logger.log_message(f"Begin inference")  # 记录开始推理
start_time = time.time()  # 记录开始时间

y_list = []
y_pred_list = []
com_coord_list = []
com_coord_pred_list = []
com_coord_per_sample_list = []

uid_list = []
smiles_list = []
sdf_name_list = []
mol_list = []
com_coord_pred_per_sample_list = []
com_coord_offset_per_sample_list = []

data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)  # 初始化进度条
for batch_id, data in enumerate(data_iter):  # 遍历数据加载器
    try:
        data = data.to(device)  # 将数据移动到设备
        LAS_tmp = []  # 初始化LAS临时变量
        for i in range(len(data)):  # 遍历数据
            LAS_tmp.append(data[i]['compound', 'LAS', 'compound'].edge_index.detach().clone())  # 获取LAS边索引
        with torch.no_grad():  # 禁用梯度计算
            com_coord_pred, compound_batch = model.inference(data)  # 调用模型推理
        post_optim_mol(args, accelerator, data, com_coord_pred, com_coord_pred_per_sample_list, com_coord_per_sample_list, compound_batch, LAS_tmp=LAS_tmp, rigid=args.rigid)  # 调用后优化函数
    except:
        continue  # 如果发生异常，跳过当前批次

if args.sdf_to_mol2:  # 如果启用了SDF到MOL2转换
    from utils.sdf_to_mol2 import convert_sdf_to_mol2  # 导入转换函数

if args.write_mol_to_file:  # 如果启用了写入分子文件
    info = pd.DataFrame({'uid': uid_list, 'smiles': smiles_list, 'sdf_name': sdf_name_list})  # 创建数据框
    info.to_csv(os.path.join(args.sdf_output_path_post_optim, f"uid_smiles_sdfname.csv"), index=False)  # 保存数据框为CSV文件
    for i in tqdm(range(len(info))):  # 遍历数据框
        save_coords = com_coord_pred_per_sample_list[i] + com_coord_offset_per_sample_list[i]  # 计算保存坐标
        sdf_output_path = os.path.join(args.sdf_output_path_post_optim, info.iloc[i]['sdf_name'])  # 定义SDF输出路径
        mol = write_mol(reference_mol=mol_list[i], coords=save_coords, output_file=sdf_output_path)  # 写入分子文件
        if args.sdf_to_mol2:  # 如果启用了SDF到MOL2转换
            convert_sdf_to_mol2(sdf_output_path, sdf_output_path.replace('.sdf', '.mol2'))  # 转换文件格式

end_time = time.time()  # 记录结束时间
logger.log_message(f"End test, time spent: {end_time - start_time}")  # 记录推理结束时间和耗时


