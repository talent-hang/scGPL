import pandas as pd
import numpy as np
import time
import random
from tqdm import tqdm
import subprocess
import os
import re
import warnings
import requests
from transformers import AutoTokenizer, AutoModel
import shutil
import json

# 数据源与输出的根路径配置
ROOT_DATA_BASE = "yourpath/Benchmark Dataset"
OUTPUT_BASE = "yourpath/data/Dataspilt"


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def Hard_Negative_Specific_train_test_val(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file,
                                          ratio=0.67, p_val=0.5):
    """
    生成硬负样本的训练、验证、测试集分割
    
    Args:
        label_file: 标签文件路径
        Gene_file: 基因文件路径
        TF_file: 转录因子文件路径
        train_set_file: 训练集输出文件路径
        val_set_file: 验证集输出文件路径
        test_set_file: 测试集输出文件路径
        ratio: 训练集比例
        p_val: 验证集概率
    """
    label = pd.read_csv(label_file, index_col=0)
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values

    tf = label['TF'].values
    tf_list = np.unique(tf).tolist()

    # 构建正样本字典
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    # 构建负样本字典
    neg_dict = {}
    for i in tf_set:
        neg_dict[i] = []

    for i in tf_set:
        if i in pos_dict.keys():
            pos_item = pos_dict[i]
            pos_item.append(i)
            pos_dict[i] = [x for x in pos_dict[i] if x != i]
            neg_item = [x for x in gene_set if x not in pos_item]
            neg_dict[i].extend(neg_item)
        else:
            neg_item = [x for x in gene_set if x != i]
            neg_dict[i].extend(neg_item)

    # 分割正样本
    train_pos = {}
    val_pos = {}
    test_pos = {}
    for k in pos_dict.keys():
        if len(pos_dict[k]) == 1:
            p = np.random.uniform(0, 1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]
        elif len(pos_dict[k]) == 2:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:int(len(pos_dict[k]) * ratio)]
            val_pos[k] = pos_dict[k][int(len(pos_dict[k]) * ratio):int(len(pos_dict[k]) * (ratio + 0.1))]
            test_pos[k] = pos_dict[k][int(len(pos_dict[k]) * (ratio + 0.1)):]

    # 分割负样本
    train_neg = {}
    val_neg = {}
    test_neg = {}
    for k in pos_dict.keys():
        neg_num = len(neg_dict[k])
        np.random.shuffle(neg_dict[k])
        train_neg[k] = neg_dict[k][:int(neg_num * ratio)]
        val_neg[k] = neg_dict[k][int(neg_num * ratio):int(neg_num * (0.1 + ratio))]
        test_neg[k] = neg_dict[k][int(neg_num * (0.1 + ratio)):]

    # 生成训练集
    train_pos_set = []
    for k in train_pos.keys():
        for val in train_pos[k]:
            train_pos_set.append([k, val])
    train_neg_set = []
    for k in train_neg.keys():
        for val in train_neg[k]:
            train_neg_set.append([k, val])
    train_set = train_pos_set + train_neg_set
    print('train pos:neg = {}:{}'.format(len(train_pos_set), len(train_neg_set)),
          round(len(train_pos_set) / len(train_neg_set), 4))
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]
    train_sample = np.array(train_set)

    train = pd.DataFrame()
    train['TF'] = train_sample[:, 0]
    train['Target'] = train_sample[:, 1]
    train['Label'] = train_label
    train.to_csv(train_set_file)

    # 生成验证集
    val_pos_set = []
    for k in val_pos.keys():
        for val in val_pos[k]:
            val_pos_set.append([k, val])
    val_neg_set = []
    for k in val_neg.keys():
        for val in val_neg[k]:
            val_neg_set.append([k, val])
    val_set = val_pos_set + val_neg_set
    print('val pos:neg = {}:{}'.format(len(val_pos_set), len(val_neg_set)),
          round(len(val_pos_set) / len(val_neg_set), 4))
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set))]
    val_sample = np.array(val_set)
    val = pd.DataFrame()
    val['TF'] = val_sample[:, 0]
    val['Target'] = val_sample[:, 1]
    val['Label'] = val_label
    val.to_csv(val_set_file)

    # 生成测试集
    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    test_neg_set = []
    for k in test_neg.keys():
        for j in test_neg[k]:
            test_neg_set.append([k, j])
    test_set = test_pos_set + test_neg_set
    print('test pos:neg = {}:{}'.format(len(test_pos_set), len(test_neg_set)),
          round(len(test_pos_set) / len(test_neg_set), 4))
    test_label = [1 for _ in range(len(test_pos_set))] + [0 for _ in range(len(test_neg_set))]
    test_sample = np.array(test_set)
    test = pd.DataFrame()
    test['TF'] = test_sample[:, 0]
    test['Target'] = test_sample[:, 1]
    test['Label'] = test_label
    test.to_csv(test_set_file)


def gen_gene_name(net_type, data_type):
    """
    生成基因名称文件
    
    Args:
        net_type: 网络类型
        data_type: 数据类型
    """
    num = [500, 1000]
    data_root_path = f'{ROOT_DATA_BASE}/{net_type}'
    for num_i in num:
        Gene2file = f'{data_root_path}/{data_type}/TFs+{num_i}/Target.csv'
        father_dir = f'{OUTPUT_BASE}/{net_type}/{data_type}/TFs_{num_i}'
        if not os.path.exists(father_dir):
            os.makedirs(father_dir)
        gene_name_file = f'{father_dir}/gene_name.txt'
        gene = pd.read_csv(Gene2file, index_col=0, header=0)
        print(len(gene['Gene']) == len(set(gene['Gene'])))
        gene['Gene'].to_csv(gene_name_file, sep='\t', index=False)
    print('gen gene name finished')


def data_split(net_type, data_type):
    """
    执行数据集分割
    
    Args:
        net_type: 网络类型
        data_type: 数据类型
    """
    num_list = [500, 1000]

    for num in num_list:
        print(f"Processing num {num}")

        dir_name = f"TFs+{num}"
        source_base = f"{ROOT_DATA_BASE}/{net_type}/{data_type}/{dir_name}"
        output_base = f"{OUTPUT_BASE}/{net_type}/{data_type}/TFs_{num}"

        os.makedirs(output_base, exist_ok=True)

        TF2file = f"{source_base}/TF.csv"
        Gene2file = f"{source_base}/Target.csv"
        label_file = f"{source_base}/Label.csv"

        print(f"Checking file existence: {label_file}")
        if not os.path.exists(label_file):
            print(f"⚠️ 警告：标签文件 {label_file} 不存在")
            continue

        train_set_file = f"{output_base}/Train_set.csv"
        val_set_file = f"{output_base}/Validation_set.csv"
        test_set_file = f"{output_base}/Test_set.csv"

        print(f"train_set_filepath: {train_set_file}")
        Hard_Negative_Specific_train_test_val(label_file, Gene2file, TF2file,
                                              train_set_file, val_set_file, test_set_file)

    print('data split finished')


def data_move(net_type, data_type):
    """
    移动数据文件到data_split文件夹
    
    Args:
        net_type: 网络类型
        data_type: 数据类型
    """
    source_folder = f'{ROOT_DATA_BASE}/{net_type}'
    destination_folder = f'{OUTPUT_BASE}/{net_type}'

    num = [500, 1000]
    files_to_copy = ['BL--ExpressionData.csv', 'BL--network.csv', 'Label.csv', 'Target.csv', 'TF.csv',
                     'Train_set.csv', 'Test_set.csv', 'Validation_set.csv']

    for num_i in num:
        for file_i in files_to_copy:
            source_file_path = f"{source_folder}/{data_type}/TFs+{num_i}/{file_i}"
            destination_file_path = f"{destination_folder}/{data_type}/TFs_{num_i}/{file_i}"
            
            # 确保目标目录存在
            dest_dir = os.path.dirname(destination_file_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            if os.path.exists(source_file_path):
                shutil.copy2(source_file_path, destination_file_path)
                print(f'copyfiles: {source_file_path} -> {destination_file_path}')
            else:
                print(f'The source file does not exist: {source_file_path}')
    print('data move finished')


def file_move(net_type, data_type):
    """
    移动所有生成的文件到指定路径
    
    Args:
        net_type: 网络类型
        data_type: 数据类型
    """
    source_folder = f'{ROOT_DATA_BASE}/{net_type}'
    destination_folder = f'{OUTPUT_BASE}/{net_type}'

    num = [500, 1000]
    raw_files = [
        'BL--ExpressionData.csv',
        'BL--network.csv',
        'Label.csv',
        'Target.csv',
        'TF.csv'
    ]

    for num_i in num:
        source_dir = f"{source_folder}/{data_type}/TFs+{num_i}"
        dest_dir = f"{destination_folder}/{data_type}/TFs_{num_i}"
        
        # 确保目标目录存在
        os.makedirs(dest_dir, exist_ok=True)

        # 拷贝原始必需文件到输出目录，避免修改源数据
        for file_i in raw_files:
            source_file_path = f"{source_dir}/{file_i}"
            destination_file_path = f"{dest_dir}/{file_i}"
            
            if os.path.exists(source_file_path):
                shutil.copy2(source_file_path, destination_file_path)
                print(f'✅ Copied raw: {source_file_path} -> {destination_file_path}')
            else:
                print(f'⚠️  Raw file not found: {source_file_path}')
    
    print(f'📁 Raw files copied to: {destination_folder}')


def causual_neg(dataset):
    """
    生成因果负样本
    
    Args:
        dataset: 输入数据集
        
    Returns:
        包含因果负样本的数据集
    """
    print('original shape', dataset.shape)
    a = dataset[dataset['Label'] == 1]
    print('positive shape', a.shape)
    a['TF_new'] = a['TF']
    a['TF'] = a['Target']
    a['Target'] = a['TF_new']
    a['Label'] = 2
    a = a.drop('TF_new', axis=1)
    dataset = pd.concat([dataset, a], axis=0)
    print('final shape', dataset.shape)
    return dataset


def casual_inference(net_type, data_type):
    """
    执行因果推理，生成因果负样本
    
    Args:
        net_type: 网络类型
        data_type: 数据类型
    """
    num = [500, 1000]

    for num_i in num:
        print('Processing num', num_i)
        dir_name = f"TFs+{num_i}"

        father_dir = f'{OUTPUT_BASE}/{net_type}/{data_type}/TFs_{num_i}'
        if not os.path.exists(father_dir):
            os.makedirs(father_dir)

        train_set_file = f'{father_dir}/Train_set.csv'
        val_set_file = f'{father_dir}/Validation_set.csv'
        test_set_file = f'{father_dir}/Test_set.csv'
        train_set = pd.read_csv(train_set_file, header=0, index_col=0)
        val_set = pd.read_csv(val_set_file, header=0, index_col=0)
        test_set = pd.read_csv(test_set_file, header=0, index_col=0)

        train_set = causual_neg(train_set)
        val_set = causual_neg(val_set)
        test_set = causual_neg(test_set)
        train_set_file_c = f'{father_dir}/Train_set_c.csv'
        val_set_file_c = f'{father_dir}/Validation_set_c.csv'
        test_set_file_c = f'{father_dir}/Test_set_c.csv'

        train_set.to_csv(train_set_file_c)
        val_set.to_csv(val_set_file_c)
        test_set.to_csv(test_set_file_c)


def count_pos_neg(csv_file):
    """
    统计正负样本数量
    
    Args:
        csv_file: CSV文件路径
        
    Returns:
        pos_count: 正样本数量
        neg_count: 负样本数量
    """
    df = pd.read_csv(csv_file, index_col=0)
    pos = df[df['Label'] == 1].shape[0]
    neg = df[df['Label'] == 0].shape[0]
    return pos, neg


def save_weight_info(net_type, data_type):
    """
    保存权重信息
    
    Args:
        net_type: 网络类型
        data_type: 数据类型
    """
    num_list = [500, 1000]
    for num in num_list:
        dir_name = f"TFs+{num}"
        base_path = f"{OUTPUT_BASE}/{net_type}/{data_type}/TFs_{num}"
        train_set_file = f"{base_path}/Train_set.csv"
        val_set_file = f"{base_path}/Validation_set.csv"
        test_set_file = f"{base_path}/Test_set.csv"

        # 检查文件是否存在
        if not all(os.path.exists(f) for f in [train_set_file, val_set_file, test_set_file]):
            print(f"⚠️  Some files missing for {net_type}/{data_type}/TFs+{num}")
            continue

        train_pos, train_neg = count_pos_neg(train_set_file)
        val_pos, val_neg = count_pos_neg(val_set_file)
        test_pos, test_neg = count_pos_neg(test_set_file)

        train_pos_weight = train_neg / train_pos if train_pos > 0 else 1.0
        val_pos_weight = val_neg / val_pos if val_pos > 0 else 1.0
        test_pos_weight = test_neg / test_pos if test_pos > 0 else 1.0

        weight_info = {
            "train": {
                "pos_count": train_pos,
                "neg_count": train_neg,
                "pos_weight": train_pos_weight
            },
            "val": {
                "pos_count": val_pos,
                "neg_count": val_neg,
                "pos_weight": val_pos_weight
            },
            "test": {
                "pos_count": test_pos,
                "neg_count": test_neg,
                "pos_weight": test_pos_weight
            }
        }

        save_dir = base_path
        os.makedirs(save_dir, exist_ok=True)

        weight_file = f"{save_dir}/weight_info.json"
        with open(weight_file, 'w') as f:
            json.dump(weight_info, f, indent=4)

        print(f"✅ Weight info saved to: {weight_file}")


def process_dataset(net_type, data_type, casual_flag=True):
    """
    处理单个数据集的完整流程
    
    Args:
        net_type: 网络类型
        data_type: 数据类型
        casual_flag: 是否执行因果推理
    """
    print(f"\n🔧 Processing {net_type} - {data_type}")

    # Step 1: 生成基因名称文件
    print('Step 1: Generating gene name files')
    gen_gene_name(net_type, data_type)

    # Step 2: 生成训练、验证、测试集
    print('Step 2: Generating training, validation, and testing sets')
    data_split(net_type, data_type)
    
    if casual_flag:
        print('Step 2.1: Performing causal inference')
        casual_inference(net_type, data_type)

    # Step 3: 保存权重信息
    print('Step 3: Saving weight information')
    save_weight_info(net_type, data_type)

    # Step 4: 移动文件到指定路径
    print('Step 4: Moving files to target directory')
    file_move(net_type, data_type)


def main():
    """主函数：执行完整的数据预处理流程"""
    # 设置随机种子
    seed = 42
    seed_everything(seed)

    # 配置参数
    casual_flag = True
    
    # 根据数据集目录结构配置
    net_types = ['Specific', 'Non-Specific', 'STRING', 'Lofgof']
    data_types = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']

    # 处理每个数据集
    for net_type in net_types:
        for data_type in data_types:
            # 检查路径是否存在
            check_path = f"{ROOT_DATA_BASE}/{net_type}/{data_type}"
            if not os.path.exists(check_path):
                print(f"⚠️  Path not found: {check_path}, skipping...")
                continue
                
            try:
                process_dataset(net_type, data_type, casual_flag)
                print(f"✅ Successfully processed {net_type} - {data_type}")
            except Exception as e:
                print(f"❌ Error processing {net_type} - {data_type}: {str(e)}")
                continue

    print("🎉 All data preprocessing completed!")


if __name__ == '__main__':
    main()
