import sys
import os
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from models.multigranularity_model import MultiGranularityModel
from models.visualization import ModelVisualizer

PATH_1 = 
PATH_2 = 

# ============================================================
# 工具类
# ============================================================
class ExperimentUtils:
    @staticmethod
    def load_sample_data(data_path: str, num_samples: int = 50) -> pd.DataFrame:
        print(f"  -> 加载数据: {data_path}")
        if not os.path.exists(data_path):
            print(f"   文件不存在: {data_path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(data_path)
            num_samples = min(len(df), num_samples)

            sampled_df = df.sample(n=num_samples, random_state=114514)
            print(f"  抽取 {num_samples} 条数据 (随机种子=114514)")
            return sampled_df
        except Exception as e:
            print(f"   读取错误: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def prepare_inputs_with_data(sampled_df, tokenizer):
        inputs_list = []
        num_cols = ['posts', 'followers', 'followings', 'level', 'member', 'reposts', 'comments', 'likes']
        cat_cols = ['credit', 'gender', 'verified', 'type']
        
        for idx, (_, row) in enumerate(sampled_df.iterrows()):
            text = str(row.get('clean_text', row.get('text', '')))
            bert_in = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors="pt")
            
            llm_text = str(row.get('llm_fact_check_reason', ''))
            llm_in = tokenizer(llm_text, max_length=128, padding='max_length', truncation=True, return_tensors="pt")
            
            num_data = [float(row.get(c, 0.0) if not pd.isna(row.get(c, 0.0)) else 0.0) for c in num_cols]
            
            cat_data = []
            for c in cat_cols:
                val = int(row.get(c, 0) if not pd.isna(row.get(c, 0)) else 0)
                val = max(0, min(val, 9999)) 
                cat_data.append(val)
            
            f_s = float(row.get('fact_score', 0.5))
            l_s = float(row.get('logic_score', 0.5))
            
            char_str = str(row.get('char_text', ''))
            try:
                if '[' in char_str: char_list = eval(char_str)
                else: char_list = list(char_str)
            except: char_list = list(char_str)
            char_ids = [min(ord(c), 10000) for c in char_list[:512]]
            char_ids += [0] * (512 - len(char_ids))
            
            input_tensors = {
                'bert_input_ids': bert_in['input_ids'],
                'bert_attention_mask': bert_in['attention_mask'],
                'char_input': torch.tensor([char_ids], dtype=torch.long),
                'numerical_features': torch.tensor([num_data], dtype=torch.float),
                'categorical_features': torch.tensor([cat_data], dtype=torch.long),
                'llm_reason_input_ids': llm_in['input_ids'],
                'llm_reason_attention_mask': llm_in['attention_mask'],
                'llm_scores': torch.tensor([[f_s, l_s]], dtype=torch.float)
            }
            
            raw_data = {
                'text': text,
                'llm_reason': llm_text,
                'stats': {k: row.get(k) for k in ['posts', 'followers', 'credit', 'verified', 'gender', 'reposts']}
            }
            
            inputs_list.append({
                'input_tensors': input_tensors,
                'raw_data': raw_data
            })
            
        return inputs_list

    @staticmethod
    def write_sample_report(raw_data, gate_dict, pred_class, save_path: str):
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*40 + "\n")
            f.write("样本数据上下文\n")
            f.write("="*40 + "\n\n")
            
            f.write("原始文本\n")
            f.write(f"{raw_data['text']}\n\n")
            
            if raw_data['llm_reason']:
                f.write("LLM 推理\n")
                f.write(f"{raw_data['llm_reason']}\n\n")
            
            f.write("结构化特征\n")
            for k, v in raw_data['stats'].items():
                f.write(f"{k}: {v}\n")
            
            f.write("\n" + "="*40 + "\n")
            f.write("模型决策权重\n")
            f.write("="*40 + "\n\n")
            
            s1_mean = np.mean(gate_dict['stage1_internal_gate'].numpy())
            s2_mean = np.mean(gate_dict['stage2_text_gate'].numpy())

# ============================================================
# 主程序
# ============================================================
def main():
    print(">>> 启动实验...")
    
    config = {
        'bert_path': os.path.join(PATH_1, "bert_chinese"),
        'data_path': os.path.join(PATH_1, 'data', 'processed', 'processed_data.csv'),
        'model_path': "/root/code/best_model.pth",
        'out_dir': os.path.join(PATH_1, 'interpretability_results_v2'),
        'num_samples': 50  # 
    }
    
    
    # Load Data
    df = ExperimentUtils.load_sample_data(config['data_path'], config['num_samples'])
    if df.empty: return
    
    # Load Tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(config['bert_path'], local_files_only=True)
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return


    if not os.path.exists(config['model_path']):
        print(f"❌ 权重文件不存在: {config['model_path']}")
        return

    try:
        state_dict = torch.load(config['model_path'], map_location='cpu')
        
        # 检测 char_vocab_size
        if 'char_embedding.weight' in state_dict:
            real_vocab_size, _ = state_dict['char_embedding.weight'].shape
            print(f"  ✅ 检测到实际字符词表大小: {real_vocab_size}")
        else:
            real_vocab_size = 10000
            
        has_stage1_weights = 'internal_fusion_gate.0.weight' in state_dict
        
    except Exception as e:
        print(f"❌ 读取权重文件出错: {e}")
        return

    try:
        model = MultiGranularityModel(
            bert_model_name=config['bert_path'],
            char_vocab_size=real_vocab_size,
            num_categorical_features=4, 
            num_numerical_features=8,
            fusion_dim=512
        )
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        model.load_state_dict(state_dict, strict=True)
            
    except Exception as e:
        print(f"❌ 加载权重出错: {e}")
        return

    model.eval()
    visualizer = ModelVisualizer(model)
    
    inputs_data_list = ExperimentUtils.prepare_inputs_with_data(df, tokenizer)
    
    skip_stage1 = not has_stage1_weights
    
    # 用于存储全局数据，生成热力图
    all_stage1_weights = []
    all_stage2_weights = []
    all_preds = []
    
    for i, item in enumerate(inputs_data_list):
        data_tensors = item['input_tensors']
        raw_data = item['raw_data']
        
            
        try:
            with torch.no_grad():
                output = model(**data_tensors)
            
            pred_class = torch.argmax(output, dim=1).item()
            all_preds.append(pred_class)
            
            gate_dict = model.get_gate_weights()

            # 收集数据用于热力图
            all_stage1_weights.append(gate_dict['stage1_internal_gate'].cpu().numpy())
            all_stage2_weights.append(gate_dict['stage2_text_gate'].cpu().numpy())

            # 为每个样本生成单独的报告
            save_dir = os.path.join(config['out_dir'], f'sample_{i+1}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            if not skip_stage1:
                visualizer.plot_stage1_internal_fusion(gate_dict['stage1_internal_gate'].numpy(), os.path.join(save_dir, "01_stage1_bert_vs_llm.svg"))
            
            visualizer.plot_stage2_global_fusion(gate_dict['stage2_text_gate'].numpy(), os.path.join(save_dir, "02_stage2_text_vs_struct.svg"), gate_dict['lambda_param'], 1-gate_dict['lambda_param'])
            visualizer.plot_mechanism_summary(os.path.join(save_dir, "03_mechanism_summary.svg"))
            
            report_path = os.path.join(save_dir, "sample_info.txt")
            ExperimentUtils.write_sample_report(raw_data, gate_dict, pred_class, report_path)
            
        except Exception as e:
            print(f"❌ Sample {i+1} 失败: {e}")

    # 🔥 生成全局热力图 
    if all_stage1_weights and not skip_stage1:
        try:
            s1_array = np.array(all_stage1_weights) 
            visualizer.plot_global_stage1_heatmap(s1_array, os.path.join(config['out_dir'], "GLOBAL_stage1_heatmap.svg"))
        except Exception as e:
            print(f"生成全局 Stage 1 热力图失败: {e}")
    
    if all_stage2_weights:
        try:
            s2_array = np.array(all_stage2_weights) 
            visualizer.plot_global_stage2_heatmap(s2_array, os.path.join(config['out_dir'], "GLOBAL_stage2_heatmap.svg"))
        except Exception as e:
            print(f"生成全局 Stage 2 热力图失败: {e}")

    print(f"结果保存在: {config['out_dir']}")

if __name__ == "__main__":
    main()
