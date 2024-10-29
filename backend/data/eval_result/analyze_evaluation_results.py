import os
import json
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt

def analyze_evaluation_results_per_file(folder_path):
    summary_data = []

    # 遍历文件夹中的所有 JSON 文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        print(f"Unsupported JSON structure in file: {file_name}")
                        continue

                    # 初始化统计结果
                    metrics = {
                        "accuracy": [],
                        "completeness": [],
                        "relevance": [],
                        "effectiveness": [],
                        "rouge1": {"precision": [], "recall": [], "fmeasure": []},
                        "rouge2": {"precision": [], "recall": [], "fmeasure": []},
                        "rougeL": {"precision": [], "recall": [], "fmeasure": []},
                        "bertscore": {"Precision": [], "Recall": [], "F1": []},
                    }

                    # 累积每个模型的数据
                    for model in data:
                        try:
                            metrics["accuracy"].append(model["accuracy"]["score"])
                            metrics["completeness"].append(model["completeness"]["score"])
                            metrics["relevance"].append(model["relevance"]["score"])
                            metrics["effectiveness"].append(model["effectiveness"]["score"])
                            
                            for rouge_key in ["rouge1", "rouge2", "rougeL"]:
                                rouge_data = model["rouge"].get(rouge_key, {})
                                metrics[rouge_key]["precision"].append(rouge_data.get("precision", 0))
                                metrics[rouge_key]["recall"].append(rouge_data.get("recall", 0))
                                metrics[rouge_key]["fmeasure"].append(rouge_data.get("fmeasure", 0))
                            
                            for bert_key in ["Precision", "Recall", "F1"]:
                                metrics["bertscore"][bert_key].append(model["bertscore"].get(bert_key, 0))
                        except KeyError as e:
                            print(f"Missing key {e} in model in file: {file_name}. Skipping this model.")
                            continue

                    # 计算平均值
                    def safe_mean(values):
                        return mean(values) if values else 0

                    # 计算总和
                    averages = {
                        "file_name": file_name,
                        "accuracy": safe_mean(metrics["accuracy"]),
                        "completeness": safe_mean(metrics["completeness"]),
                        "relevance": safe_mean(metrics["relevance"]),
                        "effectiveness": safe_mean(metrics["effectiveness"]),
                        "rouge1_fmeasure": safe_mean(metrics["rouge1"]["fmeasure"]),
                        "rouge2_fmeasure": safe_mean(metrics["rouge2"]["fmeasure"]),
                        "rougeL_fmeasure": safe_mean(metrics["rougeL"]["fmeasure"]),
                        "bertscore_F1": safe_mean(metrics["bertscore"]["F1"]),
                    }
                    averages["total_score"] = (averages["accuracy"] + averages["completeness"] +
                                               averages["relevance"] + averages["effectiveness"])

                    # 添加到汇总数据
                    summary_data.append(averages)

            except json.JSONDecodeError:
                print(f"Error reading JSON file: {file_name}")

    # 转换为 Pandas DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # 打印表格
    print("\nSummary Table:")
    print(summary_df)

    # 保存为 CSV 文件
    summary_csv_path = os.path.join(folder_path, "evaluation_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")

    # 可视化
    summary_df.set_index('file_name', inplace=True)
    summary_df.plot(kind='bar', figsize=(12, 8))
    plt.title("Comparison of Evaluation Metrics per File")
    plt.ylabel("Scores")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 使用文件夹路径调用该函数
    folder_path = "."  # 替换为实际的文件夹路径
    analyze_evaluation_results_per_file(folder_path)
