import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os


class AblationAnalyzer:
    """消融试验结果分析器"""
    
    def __init__(self, results_path, output_dir="ablation_results"):
        self.results_path = results_path
        self.output_dir = output_dir
        self.results = self.load_results()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_results(self):
        """加载结果文件"""
        with open(self.results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_summary_table(self):
        """创建结果汇总表"""
        data = []
        for config_name, result in self.results.items():
            if 'error' in result:
                continue
                
            config = result['config']
            metrics = result['test_metrics']
            
            row = {
                '模型配置': config_name,
                '描述': config['description'],
                '准确率': metrics['accuracy'],
                '精确率': metrics['precision'],
                '召回率': metrics['recall'],
                'F1分数': metrics['f1_score'],
                '使用BERT': config['use_bert'],
                '使用CNN': config['use_cnn'],
                '使用BiLSTM': config['use_bilstm'],
                '使用结构化特征': config['use_structured'],
                '使用LLM': config['use_llm'],
                '使用门控融合': config['use_gated_fusion']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 保存CSV
        csv_path = os.path.join(self.output_dir, 'ablation_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        return df
    
    def plot_performance_comparison(self, df):
        """绘制性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('消融试验性能对比', fontsize=16)
        
        # 准确率对比
        ax1 = axes[0, 0]
        sns.barplot(data=df, x='模型配置', y='准确率', ax=ax1)
        ax1.set_title('准确率对比')
        ax1.set_xlabel('模型配置')
        ax1.set_ylabel('准确率')
        ax1.tick_params(axis='x', rotation=45)
        
        # F1分数对比
        ax2 = axes[0, 1]
        sns.barplot(data=df, x='模型配置', y='F1分数', ax=ax2)
        ax2.set_title('F1分数对比')
        ax2.set_xlabel('模型配置')
        ax2.set_ylabel('F1分数')
        ax2.tick_params(axis='x', rotation=45)
        
        # 精确率和召回率对比
        ax3 = axes[1, 0]
        df_melted = pd.melt(df, id_vars=['模型配置'], 
                           value_vars=['精确率', '召回率'], 
                           var_name='指标', value_name='值')
        sns.barplot(data=df_melted, x='模型配置', y='值', hue='指标', ax=ax3)
        ax3.set_title('精确率 vs 召回率')
        ax3.set_xlabel('模型配置')
        ax3.set_ylabel('值')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        
        # 综合性能雷达图
        ax4 = axes[1, 1]
        metrics = ['准确率', '精确率', '召回率', 'F1分数']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        for idx, row in df.iterrows():
            values = [row[metric] for metric in metrics]
            values += values[:1]  # 闭合图形
            ax4.plot(angles, values, 'o-', linewidth=2, label=row['模型配置'])
            ax4.fill(angles, values, alpha=0.25)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0, 1)
        ax4.set_title('综合性能雷达图')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_component_impact(self, df):
        """分析各组件的影响"""
        # 找到完整模型的结果作为基准
        baseline = df[df['模型配置'] == 'full_model'].iloc[0]
        baseline_f1 = baseline['F1分数']
        
        component_impact = {}
        
        # 分析每个组件的影响
        components = {
            'BERT': 'use_bert',
            'CNN': 'use_cnn', 
            'BiLSTM': 'use_bilstm',
            '结构化特征': 'use_structured',
            'LLM': 'use_llm',
            '门控融合': 'use_gated_fusion'
        }
        
        for comp_name, comp_key in components.items():
            # 找到移除该组件的配置
            no_comp_configs = df[df[comp_key] == False]
            
            if not no_comp_configs.empty:
                # 计算平均性能下降
                avg_f1_drop = baseline_f1 - no_comp_configs['F1分数'].mean()
                component_impact[comp_name] = avg_f1_drop
        
        # 绘制组件影响图
        fig, ax = plt.subplots(figsize=(10, 6))
        components_list = list(component_impact.keys())
        impacts = list(component_impact.values())
        
        colors = ['red' if x > 0 else 'green' for x in impacts]
        bars = ax.bar(components_list, impacts, color=colors, alpha=0.7)
        
        ax.set_title('各组件对模型性能的影响', fontsize=14)
        ax.set_xlabel('组件')
        ax.set_ylabel('F1分数下降幅度')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for bar, impact in zip(bars, impacts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{impact:.3f}', ha='center', va='bottom' if impact > 0 else 'top')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'component_impact.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return component_impact
    
    def generate_report(self, df, component_impact):
        """生成分析报告"""
        report = f"""
# 消融试验分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 试验概述

本次消融试验共测试了 {len(df)} 种不同的模型配置，包括完整模型和各种消融版本。

## 2. 性能汇总

### 2.1 整体性能排名

"""
        
        # 按F1分数排序
        df_sorted = df.sort_values('F1分数', ascending=False)
        for idx, row in df_sorted.iterrows():
            report += f"{idx+1}. {row['模型配置']}: {row['F1分数']:.4f} ({row['描述']})\n"
        
        report += f"""
### 2.2 关键发现

- **最佳性能**: {df_sorted.iloc[0]['模型配置']} (F1: {df_sorted.iloc[0]['F1分数']:.4f})
- **完整模型性能**: {df[df['模型配置'] == 'full_model'].iloc[0]['F1分数']:.4f}
- **性能下降最大**: {df_sorted.iloc[-1]['模型配置']} (F1: {df_sorted.iloc[-1]['F1分数']:.4f})

## 3. 组件影响分析

"""
        
        # 按影响程度排序
        sorted_impacts = sorted(component_impact.items(), key=lambda x: x[1], reverse=True)
        
        for comp, impact in sorted_impacts:
            if impact > 0:
                report += f"- **{comp}**: 移除后F1分数下降 {impact:.4f}\n"
            else:
                report += f"- **{comp}**: 移除后F1分数提升 {abs(impact):.4f}\n"
        
        report += f"""
## 4. 详细结果表

{df.to_string(index=False)}

## 5. 结论与建议

基于消融试验结果，可以得出以下结论：

1. **最重要的组件**: {sorted_impacts[0][0]} (影响最大)
2. **可考虑优化的组件**: {sorted_impacts[-1][0]} (影响最小或负影响)
3. **建议的模型配置**: {df_sorted.iloc[0]['模型配置']}

## 6. 可视化图表

- 性能对比图: performance_comparison.png
- 组件影响图: component_impact.png
"""
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'ablation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"分析报告已保存到: {report_path}")
        return report
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("开始分析消融试验结果...")
        
        # 创建汇总表
        df = self.create_summary_table()
        print("✓ 汇总表已创建")
        
        # 绘制性能对比图
        self.plot_performance_comparison(df)
        print("✓ 性能对比图已生成")
        
        # 分析组件影响
        component_impact = self.analyze_component_impact(df)
        print("✓ 组件影响分析完成")
        
        # 生成报告
        report = self.generate_report(df, component_impact)
        print("✓ 分析报告已生成")
        
        print(f"\n分析完成! 所有文件保存在: {self.output_dir}")
        
        return df, component_impact, report


def main():
    """主函数"""
    # 结果文件路径
    results_path = "ablation_results/ablation_summary.json"
    
    if not os.path.exists(results_path):
        print(f"错误: 找不到结果文件 {results_path}")
        print("请先运行消融试验: python run_ablation.py")
        return
    
    # 创建分析器
    analyzer = AblationAnalyzer(results_path)
    
    # 运行分析
    df, component_impact, report = analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
