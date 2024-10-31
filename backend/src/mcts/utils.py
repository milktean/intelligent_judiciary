import os

def read_instruction(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        instruction = f.read()
    return instruction

def read_all_assistant_prompts(directory_path):
    prompts = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            strategy_name = filename[:-4]  # 去掉 .txt 后缀
            file_path = os.path.join(directory_path, filename)
            prompts[strategy_name] = read_instruction(file_path)
    return prompts

def format_conversation(conversation):
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
