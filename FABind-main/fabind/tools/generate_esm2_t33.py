import torch  # 导入 PyTorch 库，用于深度学习操作
import esm  # 导入 ESM 库，用于蛋白质语言模型
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条
import lmdb  # 导入 lmdb 库，用于处理数据库
import os  # 导入 os 库，用于文件和路径操作
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle  # 导入 pickle 库，用于序列化和反序列化数据
import sys  # 导入 sys 库，用于处理命令行参数

# 获取数据路径，命令行参数的第一个值作为基路径
data_path = os.path.join('/data3/Stone/FABind-main/FABind-main/FABind/pdbbind2020', 'dataset/processed')

# 设置 CUDA 设备为 GPU 3
device = "cuda:1" if torch.cuda.is_available() else "cpu"





# 定义氨基酸字母到数字的映射
letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                'N': 2, 'Y': 18, 'M': 12}

# 定义数字到氨基酸字母的映射
num_to_letter = {v:k for k, v in letter_to_num.items()}

# 加载预训练的 ESM2 模型和字母表
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

# 将模型加载到指定设备
model.to(device)

# 获取批量转换器，用于将序列转换为模型输入
batch_converter = alphabet.get_batch_converter()
# 模型的字母表（alphabet）会根据氨基酸序列字符串将其映射到数值化张量
model.eval()

# 打开蛋白质数据库（只读模式）
protein_db = lmdb.open(os.path.join(data_path, 'protein_1d_3d.lmdb'), readonly=True)

# 打开用于存储 ESM2 表示的数据库（写模式，设置较大的映射大小）
protein_esm2_db = lmdb.open(os.path.join(data_path, 'esm2_t33_650M_UR50D.lmdb'), map_size=40*1024 ** 3)# 1024**4

# 计算蛋白质数据库中的条目数量
with protein_db.begin(write=False) as txn:
    count = 0
    for _ in txn.cursor():
        count += 1
print(count)  # 打印条目数量

# 遍历蛋白质数据库并生成 ESM2 表示
with protein_db.begin(write=False) as txn:
    with protein_esm2_db.begin(write=True) as txn_esm2:
        cursor = txn.cursor()
        for key, value in tqdm(cursor, total=count):  # 使用进度条显示处理进度
            pdb_id = key.decode()  # 解码键（蛋白质 ID）字节类型转换为字符串类型
            seq_in_id = pickle.loads(value)[1].tolist()  # 解码值并提取蛋白质序列 将数组列表转换成字符串序列
            seq_in_str = ''.join([num_to_letter[a] for a in seq_in_id])  # 将数字序列转换为字符串序列
            # print(pdb_id,seq_in_id)
            # 准备输入数据
            data = [
                ("protein1", seq_in_str),
            ]## 蛋白质标签+蛋白质氨基酸序列
            batch_labels, batch_strs, batch_tokens = batch_converter(data)  # 转换为模型输入 标签+氨基酸序列字符串+氨基酸序列的索引张量

            # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # 计算序列长度（已注释）

            batch_tokens = batch_tokens.to(device)  # 将输入数据加载到设备
            with torch.no_grad():  # 禁用梯度计算
                results = model(batch_tokens, repr_layers=[33])  # 获取第 33 层的表示
            token_representations = results["representations"][33][0][1: -1]  # 提取表示（去掉特殊标记）
            assert token_representations.shape[0] == len(seq_in_str)  # 确保表示的长度与序列一致

            txn_esm2.put(pdb_id.encode(), pickle.dumps(token_representations.cpu()))  # 存储表示到数据库 蛋白质字节类型+pytorch张量特征

# 关闭数据库
protein_db.close()
protein_esm2_db.close()