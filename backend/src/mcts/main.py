import os
from mcts import MCTS
from model.model_factory import ModelFactory
from utils import read_instruction, read_all_assistant_prompts
from tree_saver import save_tree

def main():
    # 设置API密钥
    os.environ['DASHSCOPE_API_KEY'] = 'sk-d11a354da5f54a85a353072f24afb6e7'

    # 创建模型实例
    user_model = ModelFactory.create_model('tongyi', 'farui-plus')
    assistant_model = ModelFactory.create_model('tongyi', 'qwen-turbo')

    # 读取助理机器人的策略指令
    assistant_prompts = read_all_assistant_prompts('instructions/assistant_prompts/')

    # 初始化MCTS
    mcts = MCTS(user_model, assistant_model, assistant_prompts, max_depth=3, c_param=1.41)

    # 读取案例主题（可以从文件或其他来源读取，这里假设为示例主题）
    case_topic = "合同纠纷"  # 替换为实际的案例主题

    # 初始对话状态（空列表或预设对话开端）
    initial_state = []

    # 运行MCTS搜索
    best_action, root_node = mcts.search(initial_state, case_topic, max_iterations=100)

    # 输出最佳动作（回复）
    print("最佳回复：")
    print(f"{best_action['role']}: {best_action['content']}")

    # 保存MCTS搜索树
    save_tree(root_node, 'mcts_tree.json')

    # 可以将对话状态保存为DPO数据集的一部分
    dialogue = initial_state + [best_action]

    # 输出完整的对话
    print("\n完整的对话：")
    for msg in dialogue:
        print(f"{msg['role']}: {msg['content']}")

if __name__ == "__main__":
    main()
