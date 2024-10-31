import os

def read_instruction(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        instruction = f.read()
    return instruction

def format_conversation(conversation):
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
