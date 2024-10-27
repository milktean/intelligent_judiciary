import json
import argparse
import re
from http import HTTPStatus
from tqdm import tqdm
from bert_score import score
import dashscope
import jieba  # 导入 jieba 进行中文分词
from rouge_chinese import Rouge  # 导入 rouge-chinese 库

# 设置 Dashscope API 密钥
dashscope.api_key = ""  # 请确保安全管理您的 API 密钥

def load_file(file_path, is_json=True):
    """加载文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file) if is_json else file.read()

def save_file(data, file_path):
    """保存数据到指定文件"""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def call_dashscope_api(prompt, case):
    """调用 Dashscope API 进行评估"""
    # 替换占位符
    conversation = json.dumps(case["conversation"], ensure_ascii=False, indent=2)
    model_output = json.dumps(case["infer"], ensure_ascii=False, indent=2)
    prompt = prompt.replace("{{conversation_history}}", conversation)
    prompt = prompt.replace("{{model_output}}", model_output)
    
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    
    response = dashscope.Generation.call(
        "farui-plus",
        messages=messages,
        result_format='message'
    )
    return parse_api_response(response)

def parse_api_response(response):
    """解析 Dashscope API 响应"""
    if response.status_code == HTTPStatus.OK:
        if (response.output and
            'choices' in response.output and
            len(response.output['choices']) > 0 and
            'message' in response.output['choices'][0] and
            'content' in response.output['choices'][0]['message']):
            return response.output['choices'][0]['message']['content'].strip()
        else:
            print(f"响应结构不符合预期: {response.output}")
            return None
    else:
        print('请求失败 - Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            getattr(response, 'request_id', 'N/A'),
            getattr(response, 'status_code', 'N/A'),
            getattr(response, 'code', 'N/A'),
            getattr(response, 'message', 'N/A')
        ))
        return None

def extract_json_from_string(response_str):
    """使用正则表达式从字符串中提取 JSON"""
    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
    if json_match:
        return json_match.group()
    return None

def calculate_metrics(reference, generated):
    """计算评估指标，包括 ROUGE 和 BERTScore"""
    metrics = {}
    if reference and generated:
        metrics['rouge'] = calculate_rouge(reference, generated)
        metrics['bertscore'] = calculate_bertscore(reference, generated)
    return metrics

def tokenize(text):
    """使用 jieba 进行中文分词，并用空格连接"""
    return ' '.join(jieba.cut(text))

def remove_punctuation(text):
    """去除文本中的标点符号"""
    punctuation = '，。、；：！？（）《》“”‘’—…-'
    return ''.join(char for char in text if char not in punctuation)

def calculate_rouge(reference, generated):
    """使用 rouge-chinese 计算 ROUGE 分数，适用于中文文本"""
    # 去除标点符号
    reference = remove_punctuation(reference)
    generated = remove_punctuation(generated)
    
    # 分词
    reference = tokenize(reference)
    generated = tokenize(generated)
    
    # 初始化 ROUGE 评估器
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)
    
    # 由于 rouge.get_scores 返回的是一个列表，我们取第一个元素
    if scores:
        scores = scores[0]
    else:
        scores = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    
    return {
        "rouge1": {"precision": scores.get('rouge-1', {}).get('p', 0.0),
                   "recall": scores.get('rouge-1', {}).get('r', 0.0),
                   "fmeasure": scores.get('rouge-1', {}).get('f', 0.0)},
        "rouge2": {"precision": scores.get('rouge-2', {}).get('p', 0.0),
                   "recall": scores.get('rouge-2', {}).get('r', 0.0),
                   "fmeasure": scores.get('rouge-2', {}).get('f', 0.0)},
        "rougeL": {"precision": scores.get('rouge-l', {}).get('p', 0.0),
                   "recall": scores.get('rouge-l', {}).get('r', 0.0),
                   "fmeasure": scores.get('rouge-l', {}).get('f', 0.0)}
    }

def calculate_bertscore(reference, generated, lang='zh', model_type='bert-base-chinese'):
    """计算 BERTScore，适用于中文文本"""
    P, R, F1 = score([generated], [reference], lang=lang, model_type=model_type, rescale_with_baseline=True)
    return {
        "Precision": P.mean().item(),
        "Recall": R.mean().item(),
        "F1": F1.mean().item()
    }

def process_case(case, prompt):
    """处理单个评估案例"""
    evaluation_result_str = call_dashscope_api(prompt, case)
    if not evaluation_result_str:
        print(f"空的模型输出，跳过案例ID: {case.get('id', None)}")
        return None
    
    json_str = extract_json_from_string(evaluation_result_str)
    if not json_str:
        print(f"未找到有效的 JSON，跳过案例ID: {case.get('id', None)}")
        return None
    
    try:
        evaluation_result = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        return None
    
    # 计算平均得分
    scores = {
        "accuracy": evaluation_result.get("accuracy", {}).get("score", 0),
        "completeness": evaluation_result.get("completeness", {}).get("score", 0),
        "relevance": evaluation_result.get("relevance", {}).get("score", 0),
        "effectiveness": evaluation_result.get("effectiveness", {}).get("score", 0)
    }
    average_score = sum(scores.values()) / len(scores)
    evaluation_result["average_score"] = average_score

    # 计算其他指标（如 ROUGE 和 BERTScore）
    reference = case.get('label', {}).get('content', '').strip()
    generated = case.get('infer', {}).get('content', '').strip()
    evaluation_result.update(calculate_metrics(reference, generated))

    evaluation_result["case_id"] = case.get('id', None)
    return evaluation_result

def main():
    parser = argparse.ArgumentParser(description='批量评估脚本')
    parser.add_argument('--input', type=str, required=True, help='评测数据的输入路径')
    parser.add_argument('--output', type=str, required=True, help='评测结果的输出路径')
    parser.add_argument('--prompt', type=str, required=True, help='评估提示文件路径')
    args = parser.parse_args()

    # 加载评估提示和数据
    prompt = load_file(args.prompt, is_json=False)
    evaluation_data = load_file(args.input, is_json=True)

    results = []

    # 使用 tqdm 显示进度条处理案例
    for case in tqdm(evaluation_data, desc="评估进度"):
        result = process_case(case, prompt)
        if result:
            results.append(result)
            print(f"案例ID: {result['case_id']}, 平均得分: {result['average_score']}")
            print(f"ROUGE: {result.get('rouge', {})}")
            print(f"BERTScore: {result.get('bertscore', {})}")

    # 保存评测结果
    save_file(results, args.output)
    print(json.dumps(results, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    main()
