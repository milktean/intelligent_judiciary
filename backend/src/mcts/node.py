import math

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state          # 对话历史，列表形式
        self.parent = parent        # 父节点
        self.children = []          # 子节点列表
        self.W = 0                  # 累计奖励
        self.N = 0                  # 访问次数
        self.untried_actions = None # 未尝试的动作列表
        self.action = action        # 从父节点到当前节点的动作
        self.is_terminal = False    # 是否为终止状态

    def fully_expanded(self):
        return self.untried_actions == []

    def best_child(self, c_param=1.41):
        choices_weights = [
            (child.W / child.N) + c_param * math.sqrt(math.log(self.N) / child.N)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]
