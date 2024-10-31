class BaseModel:
    def generate_response(self, instruction, conversation, case_topic):
        """
        子类需要实现此方法，调用模型生成回复。

        :param instruction: 模型的指令（系统提示）。
        :param conversation: 对话历史，列表形式，每个元素为字典，包含 'role' 和 'content'。
        :param case_topic: 当前案例的主题，用于格式化指令。
        :return: 模型生成的回复文本，或 None（如果调用失败）。
        """
        raise NotImplementedError("子类必须实现 generate_response 方法。")

    def generate_response_from_prompt(self, prompt_content, conversation, case_topic):
        """
        根据指定的prompt内容生成回复。

        :param prompt_content: 指令的内容（字符串）。
        :param conversation: 对话历史，列表形式。
        :param case_topic: 当前案例的主题。
        :return: 模型生成的回复文本，或 None（如果调用失败）。
        """
        raise NotImplementedError("子类必须实现 generate_response_from_prompt 方法。")
