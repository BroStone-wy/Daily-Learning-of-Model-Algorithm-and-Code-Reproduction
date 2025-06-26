from torch_geometric.data import Dataset  # 导入torch_geometric的数据集基类
import pandas as pd  # 导入pandas库，用于处理数据表格
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import os  # 导入os库，用于文件路径操作
from utils.inference_pdb_utils import extract_protein_structure, extract_esm_feature  # 导入蛋白质相关的工具函数
from utils.inference_mol_utils import read_smiles, extract_torchdrug_feature_from_mol, generate_conformation  # 导入分子相关的工具函数
from torch_geometric.data import HeteroData  # 导入torch_geometric的异构数据结构
import torch  # 导入PyTorch库

class InferenceDataset(Dataset):  # 定义一个继承自Dataset的类，用于推理数据集
    def __init__(self, index_csv, pdb_file_dir, preprocess_dir):  # 初始化函数，接收索引文件路径、PDB文件目录和预处理目录
        super().__init__(None, None, None, None)  # 调用父类的初始化方法
        # 从CSV文件中提取配对索引
        with open(index_csv, 'r') as f:  # 打开CSV文件
            content = f.readlines()  # 读取所有行
        info = []  # 初始化一个空列表，用于存储信息
        for line in content[1:]:  # 跳过第一行（通常是表头），逐行处理
            smiles, pdb, ligand_id = line.strip().split(',')  # 按逗号分割每行，提取smiles、pdb和ligand_id
            info.append([smiles, pdb, ligand_id])  # 将提取的信息添加到列表中
        info = pd.DataFrame(info, columns=['smiles', 'pdb', 'ligand_id'])  # 将列表转换为pandas数据框，并指定列名

        # 读取预处理数据
        self.protein_feature, self.protein_structure = torch.load(os.path.join(preprocess_dir, 'processed_protein.pt'))  
        # 加载预处理的蛋白质特征和结构数据

        self.data = []  # 初始化一个空列表，用于存储数据
        for i in tqdm(range(len(info))):  # 使用tqdm显示进度条，遍历info中的每一行
            input_dict = {}  # 初始化一个空字典，用于存储当前样本的输入数据

            # 获取化合物信息
            try:
                smiles = info.iloc[i].smiles  # 提取当前行的smiles字符串
                mol, molecule_info = torch.load(os.path.join(preprocess_dir, 'mol', f'mol_{i}.pt'))  
                # 加载对应的分子数据和分子信息
            except:
                print('\nFailed to read molecule id ', i, ' We are skipping it.')  # 如果加载失败，打印错误信息并跳过
                continue

            # 获取蛋白质信息
            try:
                protein_structure = self.protein_structure[info.iloc[i].pdb]  # 根据PDB文件名提取蛋白质结构
                protein_esm_feature = self.protein_feature[info.iloc[i].pdb]  # 根据PDB文件名提取蛋白质ESM特征
            except:
                print('\nFailed to read protein pdb ', info.iloc[i].pdb, ' We are skipping it.')  # 如果加载失败，打印错误信息并跳过
                continue

            # 将数据添加到输入字典中
            input_dict['protein_esm_feature'] = protein_esm_feature  # 添加蛋白质ESM特征
            input_dict['protein_structure'] = protein_structure  # 添加蛋白质结构
            input_dict['molecule'] = mol  # 添加分子对象
            input_dict['molecule_smiles'] = smiles  # 添加分子SMILES字符串
            input_dict['molecule_info'] = molecule_info  # 添加分子信息
            input_dict['ligand_id'] = info.iloc[i].ligand_id  # 添加配体ID
            self.data.append(input_dict)  # 将输入字典添加到数据列表中

    def len(self):  # 定义数据集的长度方法
        return len(self.data)  # 返回数据列表的长度

    def get(self, idx):  # 定义获取单个样本的方法
        input_dict = self.data[idx]  # 根据索引获取输入字典
        protein_node_xyz = torch.tensor(input_dict['protein_structure']['coords'])[:, 1]  
        # 提取蛋白质节点的坐标
        protein_seq = input_dict['protein_structure']['seq']  # 提取蛋白质序列
        protein_esm_feature = input_dict['protein_esm_feature']  # 提取蛋白质ESM特征
        smiles = input_dict['molecule_smiles']  # 提取分子SMILES字符串
        rdkit_coords, compound_node_features, input_atom_edge_list, LAS_edge_index = input_dict['molecule_info']  
        # 提取分子信息

        n_protein_whole = protein_node_xyz.shape[0]  # 计算蛋白质节点的数量
        n_compound = compound_node_features.shape[0]  # 计算分子节点的数量

        data = HeteroData()  # 初始化一个异构数据对象

        data.coord_offset = protein_node_xyz.mean(dim=0).unsqueeze(0)  
        # 计算蛋白质节点坐标的偏移量
        protein_node_xyz = protein_node_xyz - protein_node_xyz.mean(dim=0)  
        # 将蛋白质节点坐标中心化
        coords_init = rdkit_coords - rdkit_coords.mean(axis=0)  
        # 将分子节点坐标中心化

        # compound graph
        data['compound'].node_feats = compound_node_features.float()  # 设置分子节点特征
        data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index  # 设置分子边的索引
        data['compound'].node_coords = coords_init  # 设置分子节点的初始坐标
        data['compound'].rdkit_coords = coords_init  # 设置分子节点的RDKit坐标
        data['compound'].smiles = smiles  # 设置分子的SMILES字符串
        data['compound_atom_edge_list'].x = (input_atom_edge_list[:,:2].long().contiguous() + 1).clone()  
        # 设置分子原子边列表
        data['LAS_edge_list'].x = (LAS_edge_index + 1).clone().t()  # 设置LAS边列表

        data.node_xyz_whole = protein_node_xyz  # 设置蛋白质节点的坐标
        data.seq_whole = protein_seq  # 设置蛋白质序列
        data.idx = idx  # 设置样本索引
        data.uid = input_dict['protein_structure']['name']  # 设置蛋白质结构的名称
        data.mol = input_dict['molecule']  # 设置分子对象
        data.ligand_id = input_dict['ligand_id']  # 设置配体ID
        
        # complex whole graph
        data['complex_whole_protein'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0), # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3), 
                protein_node_xyz
            ), dim=0
        ).float()
        data['complex_whole_protein'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                rdkit_coords,
                torch.zeros(1, 3), 
                torch.zeros_like(protein_node_xyz)
            ), dim=0
        ).float()

        segment = torch.zeros(n_protein_whole + n_compound + 2)
        segment[n_compound+1:] = 1 # compound: 0, protein: 1
        data['complex_whole_protein'].segment = segment # protein or ligand
        mask = torch.zeros(n_protein_whole + n_compound + 2)
        mask[:n_compound+2] = 1 # glb_p can be updated
        data['complex_whole_protein'].mask = mask.bool()
        is_global = torch.zeros(n_protein_whole + n_compound + 2)
        is_global[0] = 1
        is_global[n_compound+1] = 1
        data['complex_whole_protein'].is_global = is_global.bool()

        data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index = input_atom_edge_list[:,:2].long().t().contiguous() + 1
        data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = LAS_edge_index + 1

        data['protein_whole'].node_feats = protein_esm_feature  # 设置蛋白质的ESM特征
        
        return data  # 返回构造的异构数据对象


