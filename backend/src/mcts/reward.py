import random

def evaluate(state):
    """
    奖励函数，根据对话状态计算奖励值。
    目前返回随机数，之后可根据需求实现实际的奖励计算。

    :param state: 对话状态，列表形式
    :return: 奖励值（浮点数）
    """
    return random.uniform(0, 1)
