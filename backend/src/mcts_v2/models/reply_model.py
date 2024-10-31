from .base_model import BaseModel
from .tongyi_model import TongyiModel

class ReplyModel(BaseModel):
    def __init__(self, model_name, instruction_list):
        super().__init__(model_name, instruction_list)
        # instruction_list 是一个包含不同策略指令的列表
        self.models = [TongyiModel(model_name, instruction) for instruction in instruction_list]

    def generate_responses(self, conversation):
        # 调用不同策略的模型，生成多个回复
        responses = []
        for model in self.models:
            reply = model.generate_response(conversation)
            if reply:
                responses.append(reply)
        return responses
