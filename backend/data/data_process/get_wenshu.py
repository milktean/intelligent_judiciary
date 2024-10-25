import csv
import json
import re
import os
from tqdm import tqdm
import logging

# 配置日志记录
logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def extract_text_fields(row):
    """
    提取每行的相关字段，并构建包含元数据和正文的文本。
    假设CSV字段如下（索引从0开始）：
    0: ID
    1: 案件编号
    2: 标题
    3: 案由
    4: 当事人
    5: 法院
    6: 其他字段（如文书类型、阶段、日期等）
    11: 正文内容（索引11）
    """
    try:
        case_number = row[1]
        title = row[2]
        case_reason = row[3]
        parties = row[4]
        court = row[5]
        # 根据示例，假设正文在第12列（索引11），实际情况请调整
        text_field = row[11].strip() if len(row) > 11 else ""
    except IndexError:
        # 如果某行缺少字段，填充空值
        case_number = row[1] if len(row) > 1 else ""
        title = row[2] if len(row) > 2 else ""
        case_reason = row[3] if len(row) > 3 else ""
        parties = row[4] if len(row) > 4 else ""
        court = row[5] if len(row) > 5 else ""
        text_field = row[11].strip() if len(row) > 11 else ""
    
    # 构建元数据和正文的结合文本
    metadata = (
        f"案件编号: {case_number}\n"
        f"标题: {title}\n"
        f"案由: {case_reason}\n"
        f"当事人: {parties}\n"
        f"法院: {court}\n"
    )
    full_text = metadata + "\n" + text_field
    return full_text

def split_text(text, max_length=2000):
    """
    将文本按最大长度分割，尽量保持语义完整。
    在每个分割段落前添加元数据以保持上下文。
    """
    # 使用句子分割
    sentences = re.split(r'(?<=[。！？])', text)
    sections = []
    current_section = ""
    
    for sentence in sentences:
        if len(current_section) + len(sentence) > max_length:
            if current_section:
                sections.append(current_section.strip())
                current_section = sentence
            else:
                # 单个句子超过max_length，强制分割
                sections.append(sentence.strip())
                current_section = ""
        else:
            current_section += sentence
    
    if current_section:
        sections.append(current_section.strip())
    
    return sections

def process_row(row, max_length=2000):
    """
    处理单个CSV行，提取文本并进行必要的分割。
    返回一个列表，包含一个或多个JSON对象。
    """
    try:
        text = extract_text_fields(row)
        # 如果文本过长，进行分割
        if len(text) > max_length:
            split_sections = split_text(text, max_length)
            return [{"text": section} for section in split_sections]
        else:
            return [{"text": text}]
    except Exception as e:
        # 记录错误到日志文件
        logging.error(f"Error processing row: {e} | Row data: {row}")
        return []

def process_csv(file_path, output_path, max_length=2000, max_lines=30000000):
    """
    处理CSV文件，将每行数据转换为适用于预训练的单一JSON格式。
    使用单进程单线程，流式读取和写入，并显示处理进度。
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(file_path, 'r', encoding='utf-8') as csvfile, \
         open(output_path, 'w', encoding='utf-8') as jsonfile, \
         tqdm(total=max_lines, desc="Processing CSV") as pbar:
        
        reader = csv.reader(csvfile)
        jsonfile.write("[\n")
        first_entry = True
        processed_lines = 0
        
        for row in reader:
            if processed_lines >= max_lines:
                break
            processed = process_row(row, max_length)
            for item in processed:
                if "error" in item:
                    # 记录错误，可以选择跳过或进行其他处理
                    continue
                if not first_entry:
                    jsonfile.write(",\n")
                else:
                    first_entry = False
                jsonfile.write(json.dumps(item, ensure_ascii=False))
            pbar.update(1)
            processed_lines +=1
        
        jsonfile.write("\n]")
    
    print(f"预训练数据已保存到 {output_path}")

if __name__ == "__main__":
    # 设置CSV文件路径和输出JSON文件路径
    csv_file_path = "/home/public/liangpan/data/intelligent_judiciary/裁判文书.csv"  # 请替换为实际路径
    output_json_path = "/home/public/liangpan/data/intelligent_judiciary/pretraining_legal_documents_30B.json"
    
    # 调用处理函数
    process_csv(csv_file_path, output_json_path)
