"""
步骤1: 从数据集中抽取实验样本文本 
"""

import pandas as pd
import numpy as np
import os

#填写相应的文件读取地址，输出地址和样本输入量，并且输入随机数
RELATIVE_DATA_PATH = 
OUTPUT_DIR = 
NUM_SAMPLES = 
RANDOM_SEED = 


def get_script_dir_path(relative_path):
    """
    基于脚本所在位置的相对路径转换
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

def main():

    # 计算数据路径
    data_path = get_script_dir_path(RELATIVE_DATA_PATH)
    
    # 1. 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"\n❌ 错误: 数据文件不存在: {data_path}")
        print("请检查路径设置")
        return False
    
    
    df = pd.read_csv(data_path, encoding='utf-8')
    
    # 3. 检查必要的列
    required_columns = ['text', 'clean_text', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    
    # 4. 过滤有效数据
    valid_df = df[df['clean_text'].notna() & (df['clean_text'].str.strip() != '')].copy()
    
    num_samples = NUM_SAMPLES
    
    # 5. 随机抽样
    np.random.seed(RANDOM_SEED)
    sampled_indices = np.random.choice(valid_df.index, size=num_samples, replace=False)
    sampled_df = valid_df.loc[sampled_indices].copy().reset_index(drop=True)
    
    # 8. 保存实验样本
    output_dir_abs = get_script_dir_path(OUTPUT_DIR)
    if not os.path.exists(output_dir_abs):
        os.makedirs(output_dir_abs)
        print(f"\n创建输出目录: {output_dir_abs}")
    
    output_path = os.path.join(output_dir_abs, "experiment_samples.csv")
    sampled_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ 实验样本已保存到: {output_path}")

    
    return True

if __name__ == "__main__":
    main()
