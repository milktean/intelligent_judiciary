from .tongyi_model import TongyiModel
from .user_model import UserModel
from .reply_model import ReplyModel

def get_model(model_type, model_name, instruction):
    if model_type == 'tongyi':
        return TongyiModel(model_name, instruction)
    elif model_type == 'user':
        return UserModel(model_name, instruction)
    elif model_type == 'reply':
        return ReplyModel(model_name, instruction)
    else:
        raise ValueError(f"未知的模型类型：{model_type}")
