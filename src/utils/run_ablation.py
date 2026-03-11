import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import pickle

# 导入必要的模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import MultiGranularityDataset
from data.char_vocab import CharVocab
from data.data_preprocessing import prepare_complete_data, normalize_numerical_features
from models.ablation_model import AblationModel
from utils.ablation_config import ABLATION_CONFIGS, TRAINING_CONFIG


class AblationExperiment:
    """消融试验类"""
    
    def __init__(self, data_dir, bert_model_name, device, output_dir="ablation_results", 
                 raw_data_file="dataset2_with_llm_cleaned.csv"):
        self.data_dir = data_dir
        self.bert_model_name = bert_model_name
        self.device = device
        self.output_dir = output_dir
        self.raw_data_file = raw_data_file
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, local_files_only=True)
        
        # 检查是否需要数据划分
        self.prepare_data()
        
        # 加载vocab
        self.char_vocab = CharVocab.load(os.path.join(self.data_dir, 'char_vocab.json'))
        
        # 加载数据集
        self.load_datasets()
        
        # 存储结果
        self.results = {}
    
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.output_dir, f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        """准备数据：划分和预处理"""
        # 检查是否已经存在划分好的数据
        train_path = os.path.join(self.data_dir, 'train.csv')
        val_path = os.path.join(self.data_dir, 'val.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')
        
        if all(os.path.exists(f) for f in [train_path, val_path, test_path]):
            self.logger.info("发现已划分的数据文件，跳过数据准备步骤")
            return
        
        self.logger.info("开始数据准备和划分...")
        
        # 加载原始数据
        raw_data_path = os.path.join(self.data_dir, self.raw_data_file)
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"原始数据文件不存在: {raw_data_path}")
        
        df = pd.read_csv(raw_data_path, encoding='utf-8')
        self.logger.info(f"加载原始数据: {len(df)} 条")
        
        # 统一列名
        import re
        df.columns = [re.sub(r'\s+', '', col.strip().lower()) for col in df.columns]
        
        # 预处理
        df = prepare_complete_data(df)
        self.logger.info(f"预处理后数据: {len(df)} 条")
        
        # 构建字符词汇表
        self.logger.info("构建字符词汇表...")
        char_vocab = CharVocab(min_freq=1)
        char_vocab.build_vocab(df['char_text'].tolist())
        char_vocab.save(os.path.join(self.data_dir, 'char_vocab.json'))
        self.logger.info(f"字符词汇表大小: {len(char_vocab)}")
        
        # 数据划分
        self.logger.info("划分数据集...")
        train_val_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=42
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.125,  # 0.125 * 0.8 = 0.1 (总体的10%)
            stratify=train_val_df['label'], random_state=42
        )
        
        # 数值特征归一化
        self.logger.info("归一化数值特征...")
        train_df, val_df, test_df, scaler = normalize_numerical_features(
            train_df, val_df, test_df, 
            save_scaler_path=os.path.join(self.data_dir, 'scaler.pkl')
        )
        
        # 保存数据
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        
        self.logger.info(f"数据划分完成:")
        self.logger.info(f"训练集: {len(train_df)} 条")
        self.logger.info(f"验证集: {len(val_df)} 条")
        self.logger.info(f"测试集: {len(test_df)} 条")
    
    def load_datasets(self):
        """加载数据集"""
        self.logger.info("加载数据集...")
        
        self.train_dataset = MultiGranularityDataset(
            os.path.join(self.data_dir, 'train.csv'),
            self.tokenizer,
            self.char_vocab
        )
        
        self.val_dataset = MultiGranularityDataset(
            os.path.join(self.data_dir, 'val.csv'),
            self.tokenizer,
            self.char_vocab
        )
        
        self.test_dataset = MultiGranularityDataset(
            os.path.join(self.data_dir, 'test.csv'),
            self.tokenizer,
            self.char_vocab
        )
        
        self.logger.info(f"训练集大小: {len(self.train_dataset)}")
        self.logger.info(f"验证集大小: {len(self.val_dataset)}")
        self.logger.info(f"测试集大小: {len(self.test_dataset)}")
    
    def create_model(self, config):
        """创建模型"""
        # 🔧 更新参数列表以匹配新的消融设计
        model_params = {}
        
        # 定义AblationModel接受的参数列表
        valid_model_params = [
            'num_categorical_features', 'num_numerical_features',
            'char_embed_dim', 'cnn_out_channels', 'lstm_hidden_dim',
            'num_classes', 'dropout_rate', 'fusion_dim', 'lambda_init',
            'use_bert', 'use_cnn', 'use_bilstm', 'use_structured',
            'use_llm', 'use_llm_logic', 'use_llm_fact',  # 新增
            'use_ifg', 'use_dbg', 'use_gated_fusion'      # 新增
        ]
        
        # 只传递有效的模型参数
        for param_name in valid_model_params:
            if param_name in config:
                model_params[param_name] = config[param_name]
        
        # 🔧 确保bert_model_name被正确传递
        model = AblationModel(
            bert_model_name=self.bert_model_name,
            char_vocab_size=len(self.char_vocab),
            **model_params
        ).to(self.device)
        
        return model

    
    def create_data_loaders(self, batch_size):
        """创建数据加载器"""
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def get_optimizer(self, model):
        """获取优化器 - 兼容性处理"""
        # 方法1: 尝试从torch.optim导入 (新版本推荐)
        try:
            from torch.optim import AdamW
            optimizer = AdamW(
                model.parameters(),
                lr=TRAINING_CONFIG['learning_rate'],
                weight_decay=TRAINING_CONFIG['weight_decay']
            )
            self.logger.info("使用 torch.optim.AdamW")
            return optimizer
        except ImportError:
            pass
        
        # 方法2: 尝试从transformers导入 (旧版本)
        try:
            from transformers import AdamW
            optimizer = AdamW(
                model.parameters(),
                lr=TRAINING_CONFIG['learning_rate'],
                weight_decay=TRAINING_CONFIG['weight_decay']
            )
            self.logger.info("使用 transformers.AdamW")
            return optimizer
        except ImportError:
            pass
        
        # 方法3: 使用标准Adam作为备选
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        self.logger.warning("使用 torch.optim.Adam 作为备选")
        return optimizer
    
    def train_model(self, model, train_loader, val_loader, config_name):
        """训练模型"""
        self.logger.info(f"开始训练模型: {config_name}")
        
        # 获取优化器
        optimizer = self.get_optimizer(model)
        
        # 学习率调度器
        total_steps = len(train_loader) * TRAINING_CONFIG['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=TRAINING_CONFIG['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # 训练循环
        best_val_f1 = 0.0
        patience = 3
        patience_counter = 0
        
        for epoch in range(TRAINING_CONFIG['num_epochs']):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']} [Train]")
            for batch in train_pbar:
                # 将数据移动到设备
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(**inputs)
                
                # 计算损失
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            val_metrics = self.evaluate_model(model, val_loader)
            val_f1 = val_metrics['f1_score']
            
            self.logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val F1 = {val_f1:.4f}")
            
            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.output_dir, f"{config_name}_best_model.pth"))
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(os.path.join(self.output_dir, f"{config_name}_best_model.pth")))
        
        return model
    
    def evaluate_model(self, model, dataloader):
        """评估模型"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # 将数据移动到设备
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = model(**inputs)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_preds,
            'true_labels': all_labels
        }
    
    def run_single_experiment(self, config_name, config):
        """运行单个消融实验"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"运行消融实验: {config_name} - {config['description']}")
        self.logger.info(f"{'='*50}")
        
        # 创建模型
        model = self.create_model(config)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders(TRAINING_CONFIG['batch_size'])
        
        # 训练模型
        trained_model = self.train_model(model, train_loader, val_loader, config_name)
        
        # 在测试集上评估
        test_metrics = self.evaluate_model(trained_model, test_loader)
        
        # 保存结果
        result = {
            'config': config,
            'test_metrics': {
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1_score': test_metrics['f1_score']
            },
            'classification_report': classification_report(
                test_metrics['true_labels'], 
                test_metrics['predictions'], 
                output_dict=True
            )
        }
        
        # 保存详细结果
        result_path = os.path.join(self.output_dir, f"{config_name}_results.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"结果已保存到: {result_path}")
        self.logger.info(f"测试集准确率: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"测试集F1分数: {test_metrics['f1_score']:.4f}")
        
        return result
    
    def run_all_experiments(self):
        """运行所有消融实验"""
        self.logger.info("开始运行所有消融实验...")
        
        for config_name, config in ABLATION_CONFIGS.items():
            try:
                result = self.run_single_experiment(config_name, config)
                self.results[config_name] = result
            except Exception as e:
                self.logger.error(f"实验 {config_name} 失败: {str(e)}")
                self.results[config_name] = {'error': str(e)}
        
        # 保存汇总结果
        summary_path = os.path.join(self.output_dir, "ablation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"所有实验完成，汇总结果已保存到: {summary_path}")
        
        return self.results


def main():
    """主函数"""
    # 配置参数
    data_dir = 
    bert_model_name = 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 创建实验对象
    experiment = AblationExperiment(data_dir, bert_model_name, device)
    
    # 运行所有实验
    results = experiment.run_all_experiments()
    
    print("\n消融实验完成!")
    print(f"结果保存在: {experiment.output_dir}")


if __name__ == "__main__":
    main()
