class BaseModel:
    def __init__(self, model_name, instruction):
        self.model_name = model_name
        self.instruction = instruction

    def generate_response(self, conversation):
        raise NotImplementedError("子类需要实现 generate_response 方法")
