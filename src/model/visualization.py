import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os

class ModelVisualizer:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def plot_stage1_internal_fusion(self, gate_weights: np.ndarray, save_path: str):
        '''
        BERT vs LLM 的内部门控权重
        '''
        weights = gate_weights[0] 
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.hist(weights, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(0.5, color='red', linestyle='--', label='平衡点 (0.5)')
        ax1.set_title("Stage 1: BERT vs LLM Weight Distribution", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Gate Weight (1=Pure BERT, 0=Pure LLM)", fontsize=12)
        ax1.set_ylabel("Frequency (Feature Dimensions)", fontsize=12)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        top_llm_indices = np.argsort(weights)[:10]
        top_bert_indices = np.argsort(weights)[-10:]
        combined_indices = np.concatenate([top_llm_indices, top_bert_indices])
        combined_weights = weights[combined_indices]
        labels = [f"Dim {i}" for i in combined_indices]
        colors = ['red' if w < 0.5 else 'green' for w in combined_weights] 
        
        y_pos = np.arange(len(combined_indices))
        ax2.barh(y_pos, combined_weights, align='center', color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel("Gate Weight", fontsize=12)
        ax2.set_title("Top 20 Most Biased Dimensions", fontsize=14, fontweight='bold')
        ax2.axvline(0.5, color='black', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f" 阶段一单样本热力图已保存为SVG: {save_path}")

    def plot_stage2_global_fusion(self, gate_weights: np.ndarray, save_path: str, bert_text_contrib, struct_contrib):
        '''
        Text  Structured 的全局门控 
        '''
        weights = gate_weights[0] 
        mean_gate = np.mean(weights)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        categories = ['Super Text Feature', 'Structured Feature']
        text_dominance = mean_gate
        struct_dominance = 1 - mean_gate
        
        bars = ax.bar(categories, [text_dominance, struct_dominance], 
                      color=['#3498db', '#e74c3c'], alpha=0.8, width=0.5)
        
        ax.bar_label(bars, fmt='%.2f', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Contribution Ratio", fontsize=14)
        ax.set_title(f"Stage 2: Global Fusion Decision\nLambda: {bert_text_contrib:.3f}", 
                     fontsize=16, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        if text_dominance > 0.6:
            decision = "Decision: Relying more on TEXT content."
        elif struct_dominance > 0.6:
            decision = "Decision: Relying more on STRUCTURED data."
        else:
            decision = "Decision: Balanced fusion."
        
        ax.text(0.5, 0.95, decision, transform=ax.transAxes, ha='center', 
                fontsize=12, bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

        plt.tight_layout()
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  阶段二单样本条形图已保存为SVG: {save_path}")

    # ============================================================
    # 全局热力图函数
    # ============================================================
    
    def plot_global_stage1_heatmap(self, all_weights: np.ndarray, save_path: str):
        '''
        绘制全局 Stage 1 热力图
        '''
        if all_weights.ndim == 3:
            all_weights = np.squeeze(all_weights, axis=1)
            
        num_samples = all_weights.shape[0]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 3]})
        
        mean_weights_per_sample = np.mean(all_weights, axis=1) 
        
        ax1.plot(range(1, num_samples + 1), mean_weights_per_sample, marker='o', linestyle='-', color='#2c3e50')
        ax1.axhline(0.5, color='red', linestyle='--', label='平衡点 (0.5)')
        ax1.set_ylabel("Mean Gate Weight", fontsize=12)
        ax1.set_title(f"Stage 1: Sample-wise BERT vs LLM Policy (Total {num_samples} Samples)", fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        
        sample_weights_subset = all_weights[:, :100] 
        sns.heatmap(sample_weights_subset.T, cmap="coolwarm", center=0.5, 
                    xticklabels=5, yticklabels=10, ax=ax2, cbar_kws={'label': 'Gate Weight'})
        
        ax2.set_xlabel("Sample Index", fontsize=12)
        ax2.set_ylabel("Feature Dimension (Top 100)", fontsize=12)
        ax2.set_title("Detailed Weight Heatmap (Dim 0-99)", fontsize=12)
        
        # 
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  全局阶段一热力图已保存为SVG: {save_path}")
    
    def plot_global_stage2_heatmap(self, all_weights: np.ndarray, save_path: str):
        '''
        绘制全局 Stage 2 热力图 
        '''
        if all_weights.ndim == 3:
            all_weights = np.squeeze(all_weights, axis=1)
            
        num_samples = all_weights.shape[0]
        mean_weights_per_sample = np.mean(all_weights, axis=1) 
        
        # 创建图形，确保尺寸足够大
        fig, ax = plt.subplots(1, 1, figsize=(20, 3), facecolor='white')
        
        data_for_heatmap = mean_weights_per_sample.reshape(1, -1) 
        cmap = sns.diverging_palette(240, 10, as_cmap=True) 
        
        # 绘制热力图
        sns.heatmap(data_for_heatmap, cmap=cmap, center=0.5, 
                    xticklabels=False, yticklabels=False,  
                    vmin=0.0, vmax=1.0, ax=ax, 
                    cbar_kws={'label': 'Text Dominance Ratio'})
        
        # 手动设置x轴刻度和标签
        x_ticks = np.arange(0, num_samples, max(1, num_samples // 10))  # 每10个样本显示一个刻度
        plt.xticks(x_ticks, [str(i) for i in x_ticks], fontsize=8)
        
        # 确保x轴标签正确显示
        ax.set_xlabel("Sample Index", fontsize=12, labelpad=15)
        ax.set_yticks([])  # 隐藏y轴刻度（因为只有一行数据）
        
        # 只保留渐变色标签，移除所有其他文字
        cbar = ax.collections[0].colorbar
        cbar.set_label('Text Dominance Ratio', fontsize=12)
        
        # 移除所有标题和边框
        ax.set_title("")  # 确保没有标题
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # 调整布局，确保x轴标签不被截断
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # 为x轴标签留出更多空间
        
        plt.savefig(save_path, format='svg', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"  ✅ 全局阶段二热力图已保存为SVG: {save_path}")

    def save_all_visualizations(self, gate_dict, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        stage1_gate = gate_dict['stage1_internal_gate'].numpy()
        stage2_gate = gate_dict['stage2_text_gate'].numpy()
        lambda_val = gate_dict['lambda_param']
        
        self.plot_stage1_internal_fusion(stage1_gate, os.path.join(output_dir, "01_stage1_bert_vs_llm.svg"))
        self.plot_stage2_global_fusion(stage2_gate, os.path.join(output_dir, "02_stage2_text_vs_struct.svg"), lambda_val, 1-lambda_val)
        self.plot_mechanism_summary(os.path.join(output_dir, "03_mechanism_summary.svg"))
