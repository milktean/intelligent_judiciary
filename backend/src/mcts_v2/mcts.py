import math
import random
from models import get_model
from utils.reward import evaluate
MAX_DIALOGUE_LENGTH=4
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state          # 对话状态（列表形式）
        self.parent = parent        # 父节点
        self.children = []          # 子节点列表
        self.W = 0                  # 累计奖励
        self.N = 0                  # 访问次数
        self.untried_actions = None # 未尝试的动作列表
        self.action = action        # 导致当前节点的动作（回复）
        self.is_terminal = False    # 是否为终止状态

    def fully_expanded(self):
        return self.untried_actions == []  # 无未尝试的动作

def MCTS(root_state, user_model, reply_model, M=1000, D=10, c=1.41):
    root_node = Node(root_state)
    for m in range(M):
        node = TreePolicy(root_node, user_model, reply_model, c)
        delta = DefaultPolicy(node.state, user_model, reply_model, D)
        Backup(node, delta)
    best_child = BestChild(root_node, 0)
    return best_child.action, root_node  # 返回最佳动作和根节点

def TreePolicy(node, user_model, reply_model, c):
    while not node.is_terminal:
        if node.untried_actions is None:
            node.untried_actions = get_possible_actions(node, reply_model)
        if not node.fully_expanded():
            return Expand(node, reply_model)
        else:
            node = BestChild(node, c)
    return node

def Expand(node, reply_model):
    action = node.untried_actions.pop()
    next_state = successor_state(node.state, action)
    child_node = Node(next_state, parent=node, action=action)
    node.children.append(child_node)
    return child_node

def BestChild(node, c):
    choices_weights = [
        (child.W / child.N) + c * math.sqrt(2 * math.log(node.N) / child.N)
        for child in node.children
    ]
    return node.children[choices_weights.index(max(choices_weights))]

def DefaultPolicy(state, user_model, reply_model, D):
    depth = 0
    current_state = state.copy()
    while not is_terminal(current_state) and depth < D:
        if is_user_turn(current_state):
            user_reply = user_model.generate_response(current_state)
            current_state = successor_state(current_state, user_reply)
        else:
            # 随机选择一个回复策略
            replies = reply_model.generate_responses(current_state)
            if not replies:
                break
            reply = random.choice(replies)
            current_state = successor_state(current_state, reply)
        depth += 1
    return evaluate(current_state)

def Backup(node, delta):
    while node is not None:
        node.N += 1
        node.W += delta
        node = node.parent

def get_possible_actions(node, reply_model):
    if is_user_turn(node.state):
        # 用户机器人只有一个动作
        user_reply = node.state[-1]['content']  # 已在对话中
        return []
    else:
        # 司法问答机器人有多种策略
        replies = reply_model.generate_responses(node.state)
        return replies

def successor_state(state, action):
    new_state = state.copy()
    role = 'user' if is_user_turn(state) else 'assistant'
    new_state.append({'role': role, 'content': action})
    return new_state

def is_user_turn(state):
    return len(state) % 2 == 0  # 假设对话从用户开始

def is_terminal(state):
    # 定义终止条件，例如对话长度
    return len(state) >= MAX_DIALOGUE_LENGTH

