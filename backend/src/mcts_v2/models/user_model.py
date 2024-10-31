from .base_model import BaseModel
from .tongyi_model import TongyiModel
import random

class UserModel(BaseModel):
    def __init__(self, model_name, instruction):
        super().__init__(model_name, instruction)
        # 可以初始化用户模型所需的参数

    def generate_response(self, conversation):
        # 这里调用实际的模型API，生成用户的回复
        # 为简化，可以假设使用一个小模型或随机生成
        # 示例：
        # return "用户的提问"

        # 实际实现中，可以调用 TongyiModel 或其他模型
        # 例如：
        tongyi_model = TongyiModel(self.model_name, self.instruction)
        return tongyi_model.generate_response(conversation)
