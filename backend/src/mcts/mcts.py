import random
from node import Node
from reward import evaluate
from utils import format_conversation

class MCTS:
    def __init__(self, user_model, assistant_model, assistant_prompts, max_depth=10, c_param=1.41):
        self.user_model = user_model
        self.assistant_model = assistant_model
        self.assistant_prompts = assistant_prompts  # 策略名称到指令内容的映射
        self.max_depth = max_depth
        self.c_param = c_param

    def search(self, initial_state, case_topic, max_iterations=100):
        root_node = Node(state=initial_state)
        for _ in range(max_iterations):
            node = self.tree_policy(root_node, case_topic)
            reward = self.default_policy(node.state, case_topic)
            self.backpropagate(node, reward)
        return self.best_action(root_node), root_node  # 返回最佳动作和根节点（整个树）

    def tree_policy(self, node, case_topic):
        while not node.is_terminal and len(node.state) < self.max_depth:
            if node.untried_actions is None:
                node.untried_actions = self.get_possible_actions(node, case_topic)
            if not node.fully_expanded():
                return self.expand(node, case_topic)
            else:
                node = node.best_child(self.c_param)
        return node

    def get_possible_actions(self, node, case_topic):
        actions = []
        if len(node.state) % 2 == 0:
            # 用户的回合，只有一个动作
            action_content = self.user_model.generate_response('instructions/user_instruction.txt', node.state, case_topic)
            if action_content:
                actions.append({'role': 'user', 'content': action_content})
        else:
            # 助理的回合，有多种策略，每个策略对应一个动作
            for strategy_name, prompt in self.assistant_prompts.items():
                action_content = self.assistant_model.generate_response_from_prompt(prompt, node.state, case_topic)
                if action_content:
                    actions.append({'role': 'assistant', 'content': action_content, 'strategy': strategy_name})
        return actions

    def expand(self, node, case_topic):
        action = node.untried_actions.pop()
        new_state = node.state + [action]
        child_node = Node(state=new_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def default_policy(self, state, case_topic):
        current_state = state.copy()
        depth = len(current_state)
        while depth < self.max_depth:
            if depth % 2 == 0:
                # 用户的回合
                action_content = self.user_model.generate_response('instructions/user_instruction.txt', current_state, case_topic)
                role = 'user'
                if action_content is None:
                    break
                new_message = {'role': role, 'content': action_content}
                current_state.append(new_message)
            else:
                # 助理的回合，随机选择一个策略
                strategy_name, prompt = random.choice(list(self.assistant_prompts.items()))
                action_content = self.assistant_model.generate_response_from_prompt(prompt, current_state, case_topic)
                role = 'assistant'
                if action_content is None:
                    break
                new_message = {'role': role, 'content': action_content, 'strategy': strategy_name}
                current_state.append(new_message)
            depth += 1
        return evaluate(current_state)

    def backpropagate(self, node, reward):
        while node is not None:
            node.N += 1
            node.W += reward
            node = node.parent

    def best_action(self, root_node):
        best_child = max(root_node.children, key=lambda n: n.N)
        return best_child.action
