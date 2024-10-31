from .tongyi_model import TongyiModel

class ModelFactory:
    @staticmethod
    def create_model(model_type, model_name):
        if model_type == 'tongyi':
            return TongyiModel(model_name)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
