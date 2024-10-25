import json
import argparse
import os
import sys

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

from models.farui_model import FaruiPlusModel  # 使用绝对导入

def load_instruction(instruction_file_path):
    """加载评估指令文件内容"""
    if not os.path.exists(instruction_file_path):
        raise FileNotFoundError(f"指令文件未找到: {instruction_file_path}")
    with open(instruction_file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_test_data(input_path):
    """加载测试数据"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"测试数据文件未找到: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_results(results, output_path):
    """保存评测结果到指定路径"""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"评测结果已保存到: {output_path}")

def get_model_instance(model_name, api_key):
    """
    根据模型名称返回对应的模型实例。

    :param model_name: 模型名称，如 'farui-plus'
    :param api_key: API 密钥
    :return: 模型实例
    """
    if model_name.lower() == 'farui-plus':
        return FaruiPlusModel(api_key)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量生成司法对话的脚本')
    parser.add_argument('--input', type=str, required=True, help='测试数据的输入路径 (JSON文件)')
    parser.add_argument('--output', type=str, required=True, help='评测结果的输出路径 (JSON文件)')
    parser.add_argument('--instruction', type=str, required=True, help='评估指令文件路径 (文本文件)')
    parser.add_argument('--field', type=str, choices=['label', 'infer'], default='label', help='模型结果保存的字段 (label 或 infer)')
    parser.add_argument('--model', type=str, required=True, help='使用的模型名称，如 farui-plus')
    parser.add_argument('--api_key', type=str, required=True, help='Dashscope API 密钥')
    args = parser.parse_args()

    # 加载评估指令
    try:
        instruction = load_instruction(args.instruction)
    except FileNotFoundError as e:
        print(e)
        return

    # 加载测试数据
    try:
        test_data = load_test_data(args.input)
    except FileNotFoundError as e:
        print(e)
        return
    except json.JSONDecodeError:
        print("测试数据文件格式错误，请确保是有效的JSON文件。")
        return

    # 实例化对应的模型类
    try:
        model = get_model_instance(args.model, args.api_key)
    except ValueError as e:
        print(e)
        return

    # 遍历每个案例，调用模型生成回复并保存到指定字段
    for case in test_data:
        case_id = case.get('id')
        conversation = case.get('conversation', [])

        if not conversation:
            print(f"案例ID {case_id} 没有对话内容，跳过。")
            continue

        print(f"正在处理案例ID {case_id}...")

        # 调用模型生成回复
        model_response = model.generate_response(instruction, conversation)
        print(model_response)

        if model_response:
            case[args.field] = {
                "role": "assistant",
                "content": model_response
            }
            print(f"案例ID {case_id} 生成完成。")
        else:
            print(f"案例ID {case_id} 生成失败。")

    # 保存结果
    save_results(test_data, args.output)

    # 输出结果到控制台
    print("所有案例处理完成。")

if __name__ == '__main__':
    main()
