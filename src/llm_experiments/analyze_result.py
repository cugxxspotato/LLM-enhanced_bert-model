import pandas as pd
import numpy as np
import os

# ================= 配置参数 =================
INPUT_FILE = 
OUTPUT_DIR = 
PLOT_DIR = 
REPORT_FILE = 
# ===========================================



def get_script_dir_path(relative_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

def calculate_metrics(df):
    # ==============================
    # 1. 鲁棒性/成功率统计 
    # ==============================
    total_runs = df.groupby(['sample_id', 'prompt_type']).size().rename(('robustness', 'total_runs'))
    # 只看成功的
    df_success = df[df['success'] == True]
    success_runs = df_success.groupby(['sample_id', 'prompt_type']).size().rename(('robustness', 'success_runs'))
    
    # 构建 robust_df 
    robust_df = pd.concat([total_runs, success_runs], axis=1).fillna(0)
    robust_df[('robustness', 'failure_runs')] = robust_df[('robustness', 'total_runs')] - robust_df[('robustness', 'success_runs')]
    robust_df[('robustness', 'failure_rate')] = robust_df[('robustness', 'failure_runs')] / robust_df[('robustness', 'total_runs')]
    robust_df[('robustness', 'success_rate')] = robust_df[('robustness', 'success_runs')] / robust_df[('robustness', 'total_runs')]
    
    # ==============================
    # 2. 稳定性/标准差统计 
    # ==============================
    group_obj = df_success.groupby(['sample_id', 'prompt_type'])[['fact_score', 'logic_score']]
    stats = group_obj.agg(['mean', 'std', 'count'])
    
    # ==============================
    # 3. 数据合并 (统一结构)
    # ==============================
    full_stats = pd.concat([stats, robust_df], axis=1)
    
    # ==============================
    # 4. 计算各组平均值
    # ==============================
    group_avg = full_stats.groupby('prompt_type').mean()
    
    # ==============================
    # 5. 计算有效样本总数 
    # ==============================
    n_valid_fact = stats[stats[('fact_score', 'count')] >= 2].groupby('prompt_type').size().rename(('meta', 'valid_count_fact'))
    n_valid_logic = stats[stats[('logic_score', 'count')] >= 2].groupby('prompt_type').size().rename(('meta', 'valid_count_logic'))
    
    # 将有效样本数合并到 group_avg
    group_avg = pd.concat([group_avg, n_valid_fact, n_valid_logic], axis=1)
    
    return full_stats, group_avg

def main():
    
    input_path = get_script_dir_path(INPUT_FILE)
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到数据文件 {input_path}")
        return
    
    df = pd.read_csv(input_path)
    print(f"\n✓ 数据加载完毕: {len(df)} 条记录")
    
    df['Prompt Label'] = df['prompt_type'].map({
        'optimized_before': 'Before Optimization', 
        'optimized_after': 'After Optimization'
    })
    
    stats, group_avg = calculate_metrics(df)
    
    # --- 鲁棒性指标 ---
    success_before = group_avg.loc['optimized_before', ('robustness', 'success_rate')] * 100
    success_after = group_avg.loc['optimized_after', ('robustness', 'success_rate')] * 100
    
    # --- 样本量指标 ---
    valid_n_before = group_avg.loc['optimized_before', ('meta', 'valid_count_logic')]
    valid_n_after = group_avg.loc['optimized_after', ('meta', 'valid_count_logic')]
    
    # --- 稳定性指标  ---
    fact_std_before = group_avg.loc['optimized_before', ('fact_score', 'std')]
    logic_std_before = group_avg.loc['optimized_before', ('logic_score', 'std')]
    
    fact_std_after = group_avg.loc['optimized_after', ('fact_score', 'std')]
    logic_std_after = group_avg.loc['optimized_after', ('logic_score', 'std')]
    
    # 改善率计算 
    fact_std_improvement = 0.0
    logic_std_improvement = 0.0
    
    if fact_std_before > 0 and not pd.isna(fact_std_before):
        fact_std_improvement = (fact_std_before - fact_std_after) / fact_std_before * 100
    else:
        fact_std_improvement = np.nan
        
    if logic_std_before > 0 and not pd.isna(logic_std_before):
        logic_std_improvement = (logic_std_before - logic_std_after) / logic_std_before * 100
    else:
        logic_std_improvement = np.nan
    
    # 3. 稳定性 
    std_b_str = f"{fact_std_before:.4f}" if not pd.isna(fact_std_before) else "N/A (Samp<2)"
    std_a_str = f"{fact_std_after:.4f}" if not pd.isna(fact_std_after) else "N/A (Samp<2)"
    imp_str = f"{fact_std_improvement:+.2f}%" if not pd.isna(fact_std_improvement) else "N/A"
    
    std_b_str = f"{logic_std_before:.4f}" if not pd.isna(logic_std_before) else "N/A (Samp<2)"
    std_a_str = f"{logic_std_after:.4f}" if not pd.isna(logic_std_after) else "N/A (Samp<2)"
    imp_str = f"{logic_std_improvement:+.2f}%" if not pd.isna(logic_std_improvement) else "N/A"
 
    # ==========================================
    # 文本报告
    # ==========================================
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("LLM提示词优化对比实验分析报告 (标准差版)\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. 数据质量诊断\n")
        f.write(f"   优化前有效样本数: {valid_n_before:.0f}\n")
        f.write(f"   优化后有效样本数: {valid_n_after:.0f}\n")
        f.write("   说明: 有效样本指成功次数>=2的样本，只有这样才能计算标准差。\n\n")
        
        f.write("2. 鲁棒性分析 (成功率)\n")
        f.write(f"   优化前: {success_before:.2f}%\n")
        f.write(f"   优化后: {success_after:.2f}%\n\n")
        
        f.write("3. 稳定性分析 (标准差)\n")
        f.write(f"   逻辑评分 标准差: {logic_std_before:.4f} -> {logic_std_after:.4f}\n")
        f.write(f"   注意: 标准差数值越小，表示输出波动越小，稳定性越好。\n")
             
    print(f"  ✓ 已保存: {REPORT_FILE}")

if __name__ == "__main__":
    main()
