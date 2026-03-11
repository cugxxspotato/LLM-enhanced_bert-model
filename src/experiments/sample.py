import pandas as pd
import random

def generate_sample_data(output_path: str = "sample_data.csv", num_samples: int = 10):
    """生成示例数据用于测试"""
    # 生成随机数据
    data = {
        'bert_text': [f"这是第{i}个样本的文本内容" for i in range(num_samples)],
        'llm_text': [f"这是第{i}个样本的LLM推理文本" for i in range(num_samples)],
        'char_input': [random.randint(0, 9999) for _ in range(num_samples)],
        'numerical_features': [[random.random() for _ in range(8)] for _ in range(num_samples)],
        'categorical_features': [[random.randint(0, 9999) for _ in range(4)] for _ in range(num_samples)],
        'llm_scores': [[random.random(), random.random()] for _ in range(num_samples)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"示例数据已生成并保存到 {output_path}")

if __name__ == "__main__":
    generate_sample_data()
