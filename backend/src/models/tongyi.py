from http import HTTPStatus
import os
import dashscope
from .base_model import BaseModel


class TongyiModel(BaseModel):
    def __init__(self, model_name):
        """
        初始化 FaruiPlusModel。

        :param api_key: Dashscope API 密钥。
        """
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("环境变量 'DASHSCOPE_API_KEY' 未设置。")
        dashscope.api_key = api_key
        self.model_name = model_name


    def generate_response(self, instruction, conversation):
        """
        调用 FaruiPlus 模型生成回复。

        :param instruction: 评估指令，包含占位符的字符串。
        :param conversation: 对话历史，列表形式，每个元素为字典，包含 'role' 和 'content'。
        :return: 模型生成的回复文本，或 None（如果调用失败）。
        """
        try:
            # 填充指令中的占位符，假设占位符为 {conversation}
            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
            system_content = instruction.format(conversation=conversation_text)

            # 构建消息数组，将 instruction 作为 'system' 消息，后面跟随对话历史
            messages = [
                {'role': 'system', 'content': system_content}
            ] + conversation

            # 调用 Dashscope 的 Generation API
            response = dashscope.Generation.call(
                self.model_name,
                messages=messages,
                result_format='message',  # 设置结果为"message"格式
            )

            # 检查响应状态码
            if response.status_code == HTTPStatus.OK:
                # 打印响应数据用于调试
                print(f"响应数据: {response}")

                # 访问 'output' -> 'choices' -> 第一项 -> 'message' -> 'content'
                if (response.output and
                    'choices' in response.output and
                    len(response.output['choices']) > 0 and
                    'message' in response.output['choices'][0] and
                    'content' in response.output['choices'][0]['message']):

                    assistant_content = response.output['choices'][0]['message']['content'].strip()
                    return assistant_content
                else:
                    print(f"响应结构不符合预期: {response.output}")
                    return None
            else:
                # 打印错误信息
                print('请求失败 - Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    getattr(response, 'request_id', 'N/A'),
                    getattr(response, 'status_code', 'N/A'),
                    getattr(response, 'code', 'N/A'),
                    getattr(response, 'message', 'N/A')
                ))
                return None

        except Exception as e:
            # 捕捉并打印异常
            print(f"调用 FaruiPlus 模型时发生异常: {e}")
            return None
