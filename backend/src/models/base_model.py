from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def generate_response(self, conversation):
        """
        根据对话历史生成回复。
        
        :param conversation: 对话历史，列表形式，每个元素为字典，包含 'role' 和 'content'。
        :return: 模型生成的回复文本。
        """
        pass
