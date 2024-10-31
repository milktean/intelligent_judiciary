from models import get_model
from mcts import MCTS
import os

# 设置最大对话长度
MAX_DIALOGUE_LENGTH = 10

# 读取对话主题
def load_case_topics():
    with open('cases/case_topics.txt', 'r', encoding='utf-8') as f:
        topics = [line.strip() for line in f if line.strip()]
    return topics

def main():
    # 加载对话主题
    topics = load_case_topics()

    # 初始化用户模型和司法问答模型
    user_instruction_template = open('prompts/user_instruction.txt', 'r', encoding='utf-8').read()
    reply_instructions = [
        open('prompts/reply_instruction_strategy1.txt', 'r', encoding='utf-8').read(),
        open('prompts/reply_instruction_strategy2.txt', 'r', encoding='utf-8').read(),
        # 可以添加更多策略
    ]

    for topic in topics:
        print(f"开始处理主题：{topic}")

        # 填充用户指令
        user_instruction = user_instruction_template.format(topic=topic)
        user_model = get_model('user', 'your-user-model-name', user_instruction)

        # 初始化司法问答模型
        reply_model = get_model('reply', 'your-reply-model-name', reply_instructions)

        # 初始对话状态（空列表）
        initial_state = []

        # 执行MCTS算法
        best_action, root_node = MCTS(initial_state, user_model, reply_model, M=100, D=5, c=1.41)

        # 将搜索树保存下来
        save_search_tree(root_node, topic)

        # 构建偏好数据集（可在后续步骤中完成）
        # ...

def save_search_tree(root_node, topic):
    # 实现搜索树的保存，可以序列化为JSON格式，或绘制为图形
    # 这里简单地将搜索树打印出来
    print(f"搜索树（主题：{topic}）：")
    print_tree(root_node)

def print_tree(node, indent=''):
    print(f"{indent}{node.action} (W: {node.W}, N: {node.N})")
    for child in node.children:
        print_tree(child, indent + '    ')

if __name__ == '__main__':
    main()
