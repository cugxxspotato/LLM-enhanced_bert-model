import torch
import numpy as np
import pandas as pd
from typing import List, Dict
import os
import ast

class ExperimentUtils:
    @staticmethod
    def load_sample_data(data_path: str, num_samples: int = 10) -> pd.DataFrame:
        try:
            # 检查文件是否存在
            if not os.path.exists(data_path):
                print(f"错误: 数据文件 {data_path} 不存在")
                return pd.DataFrame()
            
            df = pd.read_csv(data_path)
            
            if len(df) < num_samples:
                print(f"警告: 数据集只有 {len(df)} 条记录，少于请求的 {num_samples} 条")
                num_samples = len(df)
            
            # 随机抽样
            sampled_df = df.sample(n=num_samples, random_state=42)
            print(f"成功从 {data_path} 抽取 {num_samples} 条样本数据")
            return sampled_df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def prepare_model_inputs(sampled_df: pd.DataFrame, 
                          bert_tokenizer, 
                          llm_tokenizer,
                          char_vocab=None) -> List[Dict]:
        """准备模型输入，根据 data_preprocessing.py 定义的列名提取数据"""
        model_inputs = []
        
        # 定义特征列名 (基于参考代码)
        # 数值特征：使用了归一化后的列名 (_norm 后缀)
        numerical_columns = [
            'posts_norm', 'followers_norm', 'followings_norm', 
            'level_norm', 'member_norm', 'reposts_norm', 
            'comments_norm', 'likes_norm'
        ]
        
        # 分类特征
        categorical_columns = ['credit', 'gender', 'verified', 'type']
        
        for _, row in sampled_df.iterrows():
            # 1. 文本特征提取
            # BERT 输入
            text = row.get('clean_text', row.get('text', ''))
            bert_inputs = bert_tokenizer(
                text, 
                padding='max_length', 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            
            # LLM 推理输入
            llm_text = row.get('llm_fact_check_reason', '')
            llm_inputs = llm_tokenizer(
                llm_text, 
                padding='max_length', 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            
            # 2. 数值特征提取
            numerical_data = []
            for col in numerical_columns:
                val = row.get(col, 0.0)
                if pd.isna(val):
                    val = 0.0
                numerical_data.append(float(val))
            
            # 添加 LLM 评分到数值特征
            fact_score = row.get('fact_score', 0.5)
            logic_score = row.get('logic_score', 0.5)
            if pd.isna(fact_score): fact_score = 0.5
            if pd.isna(logic_score): logic_score = 0.5
            numerical_data.extend([float(fact_score), float(logic_score)])
            
            # 3. 分类特征提取
            categorical_data = []
            for col in categorical_columns:
                val = row.get(col, -1)
                if pd.isna(val):
                    val = -1
                categorical_data.append(int(val))
                
            # 4. 字符特征提取

            if char_vocab:
                char_text_str = row.get('char_text', [])
                if isinstance(char_text_str, str):
                    try:
                        char_text_list = ast.literal_eval(char_text_str)
                    except:
                        char_text_list = list(char_text_str)
                else:
                    char_text_list = char_text_str if isinstance(char_text_str, list) else []
                
                char_ids = char_vocab.encode(char_text_list, max_length=512)
                char_tensor = torch.tensor(char_ids, dtype=torch.long).unsqueeze(0) # 增加 batch 维度
            else:

                raw_char = row.get('char_input', row.get('char_text', []))
                if isinstance(raw_char, str):
                     try: raw_char = ast.literal_eval(raw_char)
                     except: raw_char = []
                

                char_tensor = torch.zeros((1, 512), dtype=torch.long)

            # 转换为模型输入格式
            input_dict = {
                'bert_input_ids': bert_inputs['input_ids'],
                'bert_attention_mask': bert_inputs['attention_mask'],
                'char_input': char_tensor,
                'numerical_features': torch.tensor([numerical_data], dtype=torch.float),
                'categorical_features': torch.tensor([categorical_data], dtype=torch.long),
                'llm_reason_input_ids': llm_inputs['input_ids'],
                'llm_reason_attention_mask': llm_inputs['attention_mask'],

                'llm_scores': torch.tensor([[float(fact_score), float(logic_score)]], dtype=torch.float)
            }
            
            model_inputs.append(input_dict)
        
        return model_inputs
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """获取特征名称，对应 prepare_model_inputs 中的提取逻辑"""
        # 参考 data_preprocessing.py 和 dataset.py 中的实际列名
        feature_names = [
            # 文本特征
            'clean_text', 
            'llm_fact_check_reason', 
            'char_text', 
            # 分类特征 (4个)
            'credit', 
            'gender', 
            'verified', 
            'type', 
            # 数值特征 (8个标准特征 + 2个LLM评分)
            'posts_norm', 
            'followers_norm', 
            'followings_norm', 
            'level_norm', 
            'member_norm', 
            'reposts_norm', 
            'comments_norm', 
            'likes_norm',
            'fact_score',
            'logic_score'
        ]
        return feature_names
    
    @staticmethod
    def get_top_features(importance: np.ndarray, feature_names: List[str], top_n: int = 3) -> List[str]:
        """获取最重要的特征"""
        if len(importance) != len(feature_names):
            print(f"警告: 特征重要性长度 ({len(importance)}) 与特征名称长度 ({len(feature_names)}) 不匹配")
            # 如果不匹配，尝试根据最小长度截断或补全，但这里简单返回空或报错
            min_len = min(len(importance), len(feature_names))
            importance = importance[:min_len]
            feature_names = feature_names[:min_len]
            
        top_indices = np.argsort(importance)[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]
    
    @staticmethod
    def save_results(results: List[Dict], output_path: str = "interpretability_results.csv"):
        """保存实验结果"""
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"实验结果已保存到 {output_path}")
