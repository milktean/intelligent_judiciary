import json
import argparse
from http import HTTPStatus
import dashscope
from rouge_score import rouge_scorer
from bert_score import score

# 设置 Dashscope API 密钥
dashscope.api_key = "sk-d11a354da5f54a85a353072f24afb6e7"  # 请确保安全管理您的 API 密钥

def load_prompt(prompt_file_path):
    """加载评估提示文件内容"""
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_evaluation_data(input_path):
    """加载评估数据"""
    with open(input_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_evaluation_results(results, output_path):
    """保存评估结果到指定路径"""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

def call_with_messages(prompt, conversation):
    """调用 Dashscope API 进行评估"""
    # 替换占位符以插入具体的对话内容
    prompt_filled = prompt.replace("CONVERSATION_PLACEHOLDER", json.dumps(conversation, ensure_ascii=False))
    messages = [
        {'role': 'user', 'content': prompt_filled}
    ]
    response = dashscope.Generation.call(
        "farui-plus",
        messages=messages,
        result_format='json',  # 获取 JSON 格式的结果以便解析
    )
    if response.status_code == HTTPStatus.OK:
        return response.json()  # 假设返回的评估结果为 JSON 格式
    else:
        print('请求ID: %s, 状态码: %s, 错误代码: %s, 错误信息: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        return None

def calculate_rouge(reference, generated):
    """计算 ROUGE 分数"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    # 转换为字典格式
    return {
        "rouge1": {"precision": scores['rouge1'].precision, "recall": scores['rouge1'].recall, "fmeasure": scores['rouge1'].fmeasure},
        "rouge2": {"precision": scores['rouge2'].precision, "recall": scores['rouge2'].recall, "fmeasure": scores['rouge2'].fmeasure},
        "rougeL": {"precision": scores['rougeL'].precision, "recall": scores['rougeL'].recall, "fmeasure": scores['rougeL'].fmeasure}
    }

def calculate_bertscore(reference, generated, lang='zh'):
    """计算 BERTScore"""
    P, R, F1 = score([generated], [reference], lang=lang, rescale_with_baseline=True)
    return {
        "Precision": P.mean().item(),
        "Recall": R.mean().item(),
        "F1": F1.mean().item()
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量评估脚本')
    parser.add_argument('--input', type=str, required=True, help='评测数据的输入路径')
    parser.add_argument('--output', type=str, required=True, help='评测结果的输出路径')
    parser.add_argument('--prompt', type=str, required=True, help='评估提示文件路径')
    args = parser.parse_args()

    # 加载评估提示
    prompt = load_prompt(args.prompt)

    # 加载评估数据
    evaluation_data = load_evaluation_data(args.input)

    # 初始化结果列表
    results = []

    for case in evaluation_data:
        conversation = case.get('conversation', [])
        label = case.get('label', {})
        infer = case.get('infer', {})
        case_id = case.get('id', None)

        # 准备评估内容
        conversation_content = {
            "conversation": conversation,
            "label": label,
            "infer": infer,
            "id": case_id
        }

        # 调用模型进行评分
        evaluation_result = call_with_messages(prompt, conversation_content)
        if not evaluation_result:
            continue  # 如果评估失败，跳过此案例

        # 计算平均得分
        scores = {
            "accuracy": evaluation_result.get("accuracy", {}).get("score", 0),
            "completeness": evaluation_result.get("completeness", {}).get("score", 0),
            "relevance": evaluation_result.get("relevance", {}).get("score", 0),
            "effectiveness": evaluation_result.get("effectiveness", {}).get("score", 0)
        }
        average_score = sum(scores.values()) / len(scores)

        # 添加平均得分到结果
        evaluation_result["average_score"] = average_score

        # 检查是否存在有效的 ground truth
        ground_truth = label.get('content', '').strip()
        generated_text = infer.get('content', '').strip()
        if ground_truth and generated_text:
            rouge_scores = calculate_rouge(ground_truth, generated_text)
            bertscore_scores = calculate_bertscore(ground_truth, generated_text)
            evaluation_result["rouge"] = rouge_scores
            evaluation_result["bertscore"] = bertscore_scores

        # 添加案例ID
        evaluation_result["case_id"] = case_id

        # 添加到结果列表
        results.append(evaluation_result)

    # 保存评测结果
    save_evaluation_results(results, args.output)

    # 输出结果到控制台
    print(json.dumps(results, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    main()
