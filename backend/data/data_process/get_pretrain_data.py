import os
import glob
import json
import markdown
from bs4 import BeautifulSoup
import re

def markdown_to_text(md_content):
    """
    将 Markdown 内容转换为纯文本。
    """
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator='\n')
    return text

def split_into_sections(text):
    """
    根据章节标题（如“## 第一章 总则”）或条款（如“第一条”）拆分文本。
    """
    # 使用正则表达式匹配章节标题或条款
    sections = re.split(r'\n##\s+|第[一二三四五六七八九十]+章\s+|第[一二三四五六七八九十]+条\s+', text)
    formatted_sections = []
    for section in sections:
        section = section.strip()
        if section:
            formatted_sections.append({"text": section})
    return formatted_sections

def split_by_length(sections, max_length=500):
    """
    将分段后的文本进一步按长度限制分割，避免超出模型的上下文窗口。
    """
    final_sections = []
    for section in sections:
        text = section["text"]
        if len(text) > max_length:
            # 按句子分割，避免打断句子
            sentences = re.split(r'(?<=[。！？])', text)
            current_text = ""
            for sentence in sentences:
                if len(current_text) + len(sentence) > max_length:
                    final_sections.append({"text": current_text})
                    current_text = sentence
                else:
                    current_text += sentence
            if current_text:
                final_sections.append({"text": current_text})
        else:
            final_sections.append(section)
    return final_sections

def process_markdown_files(root_dir):
    """
    遍历根目录下的所有 Markdown 文件，转换为纯文本，分割为适当长度的段落，并存储在列表中。
    """
    data = []
    pattern = os.path.join(root_dir, '**', '*.md')
    for filepath in glob.iglob(pattern, recursive=True):
        with open(filepath, 'r', encoding='utf-8') as file:
            md_content = file.read()
            text = markdown_to_text(md_content)
            sections = split_into_sections(text)
            split_sections = split_by_length(sections)
            data.extend(split_sections)
    return data

def save_to_json(data, output_file):
    """
    将数据保存为 JSON 文件。
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    root_directory = "/home/liangpan/project/intelligent_judiciary/backend/data/raw_data/law_item/Laws"
    pretraining_data = process_markdown_files(root_directory)
    output_json = "pretraining_data.json"
    save_to_json(pretraining_data, output_json)
    print(f"预训练数据已保存到 {output_json}")
