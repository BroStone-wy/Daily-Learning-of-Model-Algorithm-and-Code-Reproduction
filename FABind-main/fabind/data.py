import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset
from utils.utils import construct_data_from_graph_gvp_mean
import lmdb
import pickle

# 定义FABindDataSet类，继承自torch_geometric.data.Dataset
class FABindDataSet(Dataset):
    def __init__(self, root, data=None, protein_dict=None, compound_dict=None, proteinMode=0, compoundMode=1,
                add_noise_to_com=None, pocket_radius=20, contactCutoff=8.0, predDis=True, args=None,
                use_whole_protein=False, compound_coords_init_mode=None, seed=42, pre=None,
                transform=None, pre_transform=None, pre_filter=None, noise_for_predicted_pocket=5.0, test_random_rotation=False, pocket_idx_no_noise=True, use_esm2_feat=False):
        # root:数据集的根目录路径。
        # data:数据集的DataFrame，包含蛋白质和化合物的相关信息。
        # protein_dict:蛋白质数据的LMDB。
        # compound_dict:化合物数据的LMDB。

        # proteinMode:蛋白质模式，0表示使用蛋白质节点坐标.
        # compoundMode:化合物模式，1表示使用化合物节点坐标。

        # add_noise_to_com:是否在训练集中添加噪声到化合物中心坐标。
        
        ### 超参数设置 ###
        # pocket_radius:口袋半径，用于计算接触点。  
        # contactCutoff:接触点的距离阈值。

        # predDis:是否预测距离分布。
        # args:其他参数。
        # use_whole_protein:是否使用整个蛋白质。
        # compound_coords_init_mode:化合物坐标初始化模式。
        
        # pre:预处理数据存储位置。
        # transform:数据变换函数。
        # pre_transform:预处理函数。
        # pre_filter:预过滤函数。
        # noise_for_predicted_pocket:预测口袋时添加的噪声。
        # test_random_rotation:测试时是否随机旋转。
        # pocket_idx_no_noise:是否在口袋索引中不添加噪声

        # 初始化数据集的参数
        self.data = data
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        # 调用父类的初始化方法
        super().__init__(root, transform, pre_transform, pre_filter)
        # print(self.processed_paths)  # 打印处理后的文件路径 它通过 root 参数和 processed_file_names 属性拼接生成。
        # 加载处理后的数据
        self.data = torch.load(self.processed_paths[0])# 配体和蛋白质数据集的元信息，用于筛选和索引。
        self.compound_rdkit_coords = torch.load(self.processed_paths[3]) # 配体初始坐标数据

        # 打开LMDB数据库以读取蛋白质和化合物数据
        self.protein_dict = lmdb.open(self.processed_paths[1], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False) # 蛋白质节点特征和坐标
        self.compound_dict = lmdb.open(self.processed_paths[2], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)# 存储化合物的节点特征，化合物坐标、边信息和距离分布；键：化合物名称；值：六元组
        # 如果使用ESM2特征，则打开相关数据库
        if use_esm2_feat:
            self.protein_esm2_feat = lmdb.open(self.processed_paths[4], readonly=True, max_readers=1, lock=False, readahead=False, meminit=False)
        # 初始化其他参数
        self.compound_coords_init_mode = compound_coords_init_mode
        self.add_noise_to_com = add_noise_to_com
        self.noise_for_predicted_pocket = noise_for_predicted_pocket
        self.proteinMode = proteinMode
        self.compoundMode = compoundMode
        self.pocket_radius = pocket_radius
        self.contactCutoff = contactCutoff
        self.predDis = predDis
        self.use_whole_protein = use_whole_protein
        self.test_random_rotation = test_random_rotation
        self.pocket_idx_no_noise = pocket_idx_no_noise
        self.use_esm2_feat = use_esm2_feat
        self.seed = seed
        self.args = args
        self.pre = pre
    
    @property
    def processed_file_names(self):
        # 定义处理后的文件名列表
        return ['data.pt', 'protein_1d_3d.lmdb', 'compound_LAS_edge_index.lmdb', 'compound_rdkit_coords.pt', 'esm2_t33_650M_UR50D.lmdb']

    def len(self):
        # 返回数据集的长度
        return len(self.data)

    def get(self, idx):
        # 根据索引获取数据
        line = self.data.iloc[idx]  # 根据索引获取数据 
        pocket_com = line['pocket_com']  # 获取口袋中心坐标
        use_compound_com = line['use_compound_com']  # 是否使用配体中心
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else self.use_whole_protein  # 元信息line是否使用整个蛋白质或者初始化蛋白质elf.use_whole_protein
        group = line['group'] if "group" in line.index else 'train'  # 数据group分组（训练、验证或测试） ，无group则使用默认值train
        # 根据分组设置噪声参数
        if group == 'train' and use_compound_com:
            add_noise_to_com = self.add_noise_to_com
        elif group == 'train' and not use_compound_com:
            add_noise_to_com = self.noise_for_predicted_pocket
        else:
            add_noise_to_com = None

        # 根据分组设置是否随机旋转
        if group == 'train':
            random_rotation = True
        elif group == 'test' and self.test_random_rotation:
            random_rotation = True
        else:
            random_rotation = False

        protein_name = line['protein_name']  # 获取蛋白质名称（PDB ID）
        if self.proteinMode == 0:
            # 从LMDB数据库中读取蛋白质节点坐标和序列
            with self.protein_dict.begin() as txn:
                protein_node_xyz, protein_seq = pickle.loads(txn.get(protein_name.encode()))
            # 如果使用ESM2特征，则读取相关数据
            if self.use_esm2_feat:
                with self.protein_esm2_feat.begin() as txn:
                    protein_esm2_feat = pickle.loads(txn.get(protein_name.encode()))
            else:
                protein_esm2_feat = None

        name = line['compound_name']  # 获取化合物名称
        rdkit_coords = self.compound_rdkit_coords[name]  # 获取化合物的RDKit坐标

        # 从LMDB数据库中读取化合物相关数据
        with self.compound_dict.begin() as txn:
            coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution, LAS_edge_index = pickle.loads(txn.get(name.encode()))
            # coords：配体的真实节点三维坐标;
            # compound_node_features：化合物节点特征，表示化合物中每个原子的属性，例如原子类型、电荷、键合信息等低维特征。
            # input_atom_edge_list：化合物原子之间的边索引列表;
            # input_atom_edge_attr_list：配体原子之间的边属性列表;表示化合物中每条边的属性；例如键类型（单键、双键等）、键强度等。
            # pair_dis_distribution：化合物中原子对之间的距离分布;
            # LAS_edge_index：化合物的LAS边索引列表，表示化合物中原子之间的局部加权边索引。


        if self.proteinMode == 0:
            # 构建图数据
            data, input_node_list, keepNode = construct_data_from_graph_gvp_mean(self.args, protein_node_xyz, protein_seq, 
                                coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, LAS_edge_index, rdkit_coords, compound_coords_init_mode=self.compound_coords_init_mode, contactCutoff=self.contactCutoff, includeDisMap=self.predDis,
                                pocket_radius=self.pocket_radius, add_noise_to_com=add_noise_to_com, use_whole_protein=use_whole_protein, pdb_id=name, group=group, seed=self.seed, data_path=self.pre, 
                                use_compound_com_as_pocket=use_compound_com, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode, random_rotation=random_rotation, pocket_idx_no_noise=self.pocket_idx_no_noise,
                                protein_esm2_feat=protein_esm2_feat)

        # 设置数据的PDB ID和分组
        data.pdb = line['pdb'] if "pdb" in line.index else f'smiles_{idx}'
        data.group = group

        return data

# 定义函数以加载数据集
def get_data(args, logger, addNoise=None, use_whole_protein=False, compound_coords_init_mode='pocket_center_rdkit', pre="./FABind/pdbbind2020"):
    if args.data == "0":
        # 记录日志信息
        logger.log_message(f"Loading dataset")
        logger.log_message(f"compound feature based on torchdrug")
        logger.log_message(f"protein feature based on esm2")
        add_noise_to_com = float(addNoise) if addNoise else None

        # 创建FABindDataSet实例
        new_dataset = FABindDataSet(f"{pre}/dataset", add_noise_to_com=add_noise_to_com, use_whole_protein=use_whole_protein, compound_coords_init_mode=compound_coords_init_mode, pocket_radius=args.pocket_radius, noise_for_predicted_pocket=args.noise_for_predicted_pocket, 
                                    test_random_rotation=args.test_random_rotation, pocket_idx_no_noise=args.pocket_idx_no_noise, use_esm2_feat=args.use_esm2_feat, seed=args.seed, pre=pre, args=args)
        # data from 数据路径预处理后的数据文件
        # 筛选训练数据
        train_tmp = new_dataset.data.query("c_length < 100 and native_num_contact > 5 and group =='train' and use_compound_com").reset_index(drop=True)
        
        # 筛选条件包括化合物中心通常用于定义化合物的初始坐标。
        # 重置筛选后的 DataFrame 的索引，确保数据集的索引是连续的，方便后续操作。

        valid_test_tmp = new_dataset.data.query("(group == 'valid' or group == 'test') and use_compound_com").reset_index(drop=True)
        new_dataset.data = pd.concat([train_tmp, valid_test_tmp], axis=0).reset_index(drop=True)
        d = new_dataset.data
        # 它包含所有样本的元信息，用于描述化合物和蛋白质的相关属性以及分组信息。
        # data的列由FABind数据类确定的，无直接具体的原子三维坐标信息，列主要用于描述化合物和蛋白质的相关属性，以及分组信息等。包含每个样本的元信息，例如化合物长度、接触数量、分组信息、蛋白质名称、化合物名称、口袋中心坐标等。是否启用了化合物中心（初始化坐标用）

        # 获取训练、验证和测试数据的索引，并包含具体的图数据。
        only_native_train_index = d.query("group =='train'").index.values
        train = new_dataset[only_native_train_index]
        valid_index = d.query("group =='valid'").index.values
        valid = new_dataset[valid_index]
        test_index = d.query("group =='test'").index.values
        test = new_dataset[test_index]

    return train, valid, test