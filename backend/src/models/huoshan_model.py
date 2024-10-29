from http import HTTPStatus
import os
from openai import OpenAI
from .base_model import BaseModel


class HuoshanModel(BaseModel):
    def __init__(self, model_name):
        """
        初始化 HuoshanModel。

        :param model_name: 火山模型的名称或端点ID。
        """
        api_key = os.getenv('ARK_API_KEY')
        if not api_key:
            raise ValueError("环境变量 'ARK_API_KEY' 未设置。")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        self.model_name = model_name

    def generate_response(self, instruction, conversation):
        """
        调用火山模型生成回复。

        :param instruction: 评估指令，包含占位符的字符串。
        :param conversation: 对话历史，列表形式，每个元素为字典，包含 'role' 和 'content'。
        :return: 模型生成的回复文本，或 None（如果调用失败）。
        """
        try:
            # 填充指令中的占位符，假设占位符为 {conversation}
            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
            system_content = instruction.format(conversation=conversation_text)

            # 构建消息数组，将填充后的 instruction 作为 'system' 消息，后面跟随对话历史
            messages = [
                {'role': 'system', 'content': system_content}
            ] + conversation

            # 调用火山模型的 Completion API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )

            # 打印响应数据用于调试
            # print(f"响应数据: {response}")

            # 检查响应对象，提取生成的内容
            if (hasattr(response, 'choices') and
                    len(response.choices) > 0 and
                    hasattr(response.choices[0], 'message') and
                    hasattr(response.choices[0].message, 'content')):

                assistant_content = response.choices[0].message.content.strip()
                return assistant_content
            else:
                print(f"响应结构不符合预期: {response}")
                return None

        except Exception as e:
            # 捕捉并打印异常
            print(f"调用火山模型时发生异常: {e}")
            return None
