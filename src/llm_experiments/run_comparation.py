import pandas as pd
import numpy as np
import os
import time
import json
import re
import zhipuai

# ================= 配置参数 =================
# 输入文件
INPUT_FILE = 
# 输出目录
OUTPUT_DIR = 
OUTPUT_FILENAME = 

# API设置
MODEL_NAME =                  #模型名字
MAX_RETRIES =               #失败重试次数
RETRY_DELAY =                #失败后重试等待时间(秒)
API_CALL_DELAY =          # 正常请求间隔(秒), 设为X秒防止触发频率限制

# 🔧 调试模式: True=只处理1个样本测试, False=处理全部100个样本
DEBUG_MODE = False
# ===========================================

# --- 提示词定义 ---

# 1. 优化前提示词 (简单、格式不明确、易产生幻觉)
ORIGINAL_PROMPT = """
请分析以下文本，判断其真实性，并给出评分。

文本内容：
---
{text}
---

请回答：
1. 这个文本说了什么？
2. 你认为它真实吗？
3. 事实校验评分（0-1之间）
4. 逻辑一致性评分（0-1之间）
"""

# 2. 优化后提示词 (当前使用的提示词: 格式明确、严格约束)
OPTIMIZED_PROMPT = """
你是一个专业的事实核查分析师。请对以下文本进行深入分析，并给出评分。

请从以下几个方面进行分析：
1.  **关键事实陈述**：识别文本中所有可以被验证的事实性声明。
2.  **信息来源与可信度**：评估文本中提及或暗示的信息来源的可靠性。如果文本未提及来源，请指出这一点。
3.  **语言与情感色彩**：分析文本是否使用了夸张、煽动性、情绪化或带有偏见的语言。
4.  **背景知识**：基于你的知识库，提供与文本内容相关的背景信息，这些信息可能有助于验证其真伪。

**重要约束：**
- 请不要在你的回答中使用"真实的"、"虚假的"、"可信的"、"不可信的"等任何形式的最终判断词。
- 只需客观、中立地呈现你的分析过程和发现。
- **请将分析内容严格控制在300字以内。**
- 如果内容需要更详细说明，请优先保证评分部分的完整性

**评分要求：**
在分析结束后，请提供两个0-1之间的评分：
1. 事实校验评分：评估文本内容与公开事实/常识的一致性（0表示完全不一致，1表示完全一致）
2. 逻辑一致性评分：评估文本内部逻辑是否自洽（0表示完全矛盾，1表示完全自洽）

**强制格式要求：**
1. 必须完整输出以下三个部分，缺一不可
2. 评分必须是0-1之间的纯数字
3. 输出格式必须严格遵循：
分析内容：[你的分析内容]
事实校验评分：[0-1之间的数值]
逻辑一致性评分：[0-1之间的数值]

文本内容：
---
{text}
---

请开始你的分析和评分：
"""

def get_script_dir_path(relative_path):
    """获取相对于脚本所在目录的绝对路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

def parse_response(response_text, prompt_type):
    """
    解析LLM响应
    """
    result = {
        'fact_score': 0.5,
        'logic_score': 0.5,
        'analysis': response_text[:500],
        'success': True
    }
    
    if prompt_type == 'optimized':
        # 优化提示词有明确的格式，用正则提取
        analysis_match = re.search(
            r'分析内容[:：]\s*(.*?)(?=\n事实校验评分|$)', 
            response_text, 
            re.DOTALL
        )
        fact_match = re.search(r'事实校验评分[:：]\s*([0-9.]+)', response_text)
        logic_match = re.search(r'逻辑一致性评分[:：]\s*([0-9.]+)', response_text)
        
        # 兜底匹配：如果标准位置找不到，尝试在文本末尾找
        if not (fact_match and logic_match):
            last_part = response_text[-150:]
            fact_match = fact_match or re.search(r'事实校验评分[:：]\s*([0-9.]+)', last_part)
            logic_match = logic_match or re.search(r'逻辑一致性评分[:：]\s*([0-9.]+)', last_part)
        
        if analysis_match:
            result['analysis'] = analysis_match.group(1).strip()
        if fact_match:
            try:
                result['fact_score'] = float(fact_match.group(1))
            except:
                pass
        if logic_match:
            try:
                result['logic_score'] = float(logic_match.group(1))
            except:
                pass
    
    else:
        # 原始提示词格式不明确，尝试从文本中提取数字
        # 寻找两个0-1之间的数字
        scores = re.findall(r'[0-9]*\.?[0-9]+', response_text)
        scores = [float(s) for s in scores if 0 <= float(s) <= 1]
        
        if len(scores) >= 2:
            result['fact_score'] = scores[0]
            result['logic_score'] = scores[1]
            # 移除评分部分，保留分析内容
            analysis = re.sub(r'[0-9]*\.?[0-9]+', '', response_text)
            result['analysis'] = analysis[:500]
        elif len(scores) == 1:
            result['fact_score'] = scores[0]
            result['logic_score'] = 0.5
        else:
            # 如果找不到评分，标记为失败
            result['success'] = False
    
    # 归一化评分
    result['fact_score'] = max(0.0, min(1.0, result['fact_score']))
    result['logic_score'] = max(0.0, min(1.0, result['logic_score']))
    
    return result

def call_llm_single(text, prompt_template, prompt_type, client):
    """
    单次LLM调用
    """
    prompt = prompt_template.format(text=text)
    
    for attempt in range(MAX_RETRIES):
        try:
            # 不打印每一步，避免刷屏，在主函数打印
            start_time = time.time()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个文本分析助手。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=600,
                tools=[{"type": "web_search", "web_search": {"enable": False}}],
            )
            
            elapsed = time.time() - start_time
            
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise Exception("API返回无效响应")
            
            response_text = response.choices[0].message.content.strip()
            
            # 解析响应
            parsed = parse_response(response_text, prompt_type)
            parsed['raw_response'] = response_text
            
            return parsed
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                # 返回错误信息
                return {
                    'fact_score': 0.5,
                    'logic_score': 0.5,
                    'analysis': f"Error: {str(e)}",
                    'raw_response': "",
                    'success': False
                }

def main():
    
    # 调试模式提示
    if DEBUG_MODE:
        print("\n⚠️  调试模式已开启：仅处理1个样本，每次请求间隔3秒\n")
    else:
        print("\n🚀 正式模式：处理100个样本，每次请求间隔3秒\n")
    
    # 1. 初始化客户端
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        print("❌ 错误: 未设置环境变量 ZHIPUAI_API_KEY")
        return
    client = zhipuai.ZhipuAI(api_key=api_key)
    print(f"✓ API客户端初始化成功 (模型: {MODEL_NAME})")
    
    # 2. 加载样本
    input_path = get_script_dir_path(INPUT_FILE)
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到样本文件 {input_path}")
        print("请确保步骤1已成功运行")
        return
    
    df_samples = pd.read_csv(input_path)
    print(f"✓ 加载样本数: {len(df_samples)}")
    
    if DEBUG_MODE:
        df_samples = df_samples.head(1)
        print(f"✓ 调试模式：只处理前1条样本")
    
    # 3. 准备结果存储
    all_results = []
    
    
    for sample_idx in range(len(df_samples)):
        row = df_samples.iloc[sample_idx]
        text = row['clean_text']
        sample_id = row.get('id', sample_idx)
        
        # 手动打印进度
        print(f">>> 正在处理样本 {sample_idx + 1}/{len(df_samples)} (ID: {sample_id})")
        print(f"    文本长度: {len(text)} 字符")
        
        for run_idx in range(10):
            # --- 优化前的提示词 ---
            print(f"    Run {run_idx + 1}/20 (优化前)...", end='')
            
            res_orig = call_llm_single(text, ORIGINAL_PROMPT, 'original', client)
            
            # 打印结果：成功显示分数，失败显示错误信息
            if res_orig['success']:
                print(f" ✅ (F:{res_orig['fact_score']}, L:{res_orig['logic_score']})")
            else:
                # 截取前50个字符的错误信息
                error_msg = res_orig['analysis'][:50]
                print(f" ❌ {error_msg}...")
            
            all_results.append({
                'sample_id': sample_id,
                'text': text,
                'label': row['label'],
                'prompt_type': 'optimized_before',
                'run_id': run_idx,
                'fact_score': res_orig['fact_score'],
                'logic_score': res_orig['logic_score'],
                'analysis': res_orig['analysis'],
                'success': res_orig['success']
            })
            time.sleep(API_CALL_DELAY) # 等待 3 秒
            
            # --- 优化后的提示词 ---
            print(f"    Run {run_idx + 1}/20 (优化后)...", end='')
            res_opt = call_llm_single(text, OPTIMIZED_PROMPT, 'optimized', client)
            
            if res_opt['success']:
                print(f" ✅ (F:{res_opt['fact_score']}, L:{res_opt['logic_score']})")
            else:
                error_msg = res_opt['analysis'][:50]
                print(f" ❌ {error_msg}...")
            
            all_results.append({
                'sample_id': sample_id,
                'text': text,
                'label': row['label'],
                'prompt_type': 'optimized_after',
                'run_id': run_idx,
                'fact_score': res_opt['fact_score'],
                'logic_score': res_opt['logic_score'],
                'analysis': res_opt['analysis'],
                'success': res_opt['success']
            })
            time.sleep(API_CALL_DELAY) # 等待 3 秒
        
        # 每处理完一个样本，保存一次
        output_dir = get_script_dir_path(OUTPUT_DIR)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        temp_output_path = os.path.join(output_dir, f"temp_{OUTPUT_FILENAME}")
        pd.DataFrame(all_results).to_csv(temp_output_path, index=False, encoding='utf-8')
    
    # 5. 最终保存
    final_output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(final_output_path, index=False, encoding='utf-8')
    
    print("\n" + "=" * 60)
    print("步骤2完成!")
    print("=" * 60)
    print(f"✓ 总处理记录数: {len(df_results)}")
    print(f"✓ 结果文件: {final_output_path}")
    
    # 简单统计
    before_success = df_results[(df_results['prompt_type'] == 'optimized_before') & (df_results['success'])].shape[0]
    after_success = df_results[(df_results['prompt_type'] == 'optimized_after') & (df_results['success'])].shape[0]
    print(f"\n成功率统计:")
    print(f"  优化前提示词: {before_success}/1000")
    print(f"  优化后提示词: {after_success}/1000")
    
    print("\n下一步: 运行步骤3 - 结果分析与可视化")
    print("=" * 60)

if __name__ == "__main__":
    main()
