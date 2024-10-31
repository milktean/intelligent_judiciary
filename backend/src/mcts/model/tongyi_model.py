from http import HTTPStatus
import os
import dashscope
from .base_model import BaseModel
from utils import format_conversation

class TongyiModel(BaseModel):
    def __init__(self, model_name):
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("环境变量 'DASHSCOPE_API_KEY' 未设置。")
        dashscope.api_key = api_key
        self.model_name = model_name

    def generate_response(self, instruction, conversation, case_topic):
        try:
            # 读取并格式化指令
            with open(instruction, 'r', encoding='utf-8') as f:
                system_content = f.read()
            system_content = system_content.format(conversation=conversation, case_topic=case_topic)


            # 构建消息数组，将 instruction 作为 'system' 消息，后面跟随对话历史
            messages = [
                {'role': 'user', 'content': system_content}
            ]
            print(messages)

            # 调用 Dashscope 的 Generation API
            response = dashscope.Generation.call(
                self.model_name,
                messages=messages,
                result_format='message',  # 设置结果为"message"格式
            )

            # 检查响应状态码
            if response.status_code == HTTPStatus.OK:
                # 访问 'output' -> 'choices' -> 第一项 -> 'message' -> 'content'
                if (response.output and
                    'choices' in response.output and
                    len(response.output['choices']) > 0 and
                    'message' in response.output['choices'][0] and
                    'content' in response.output['choices'][0]['message']):

                    assistant_content = response.output['choices'][0]['message']['content'].strip()
                    print(assistant_content)
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
            print(f"调用模型时发生异常: {e}")
            return None
    def generate_response_from_prompt(self, prompt_content, conversation, case_topic):
        try:
            # 格式化指令
            content = prompt_content.format(conversation=format_conversation(conversation), case_topic=case_topic)

            # 构建消息数组，将 instruction 作为 'system' 消息，后面跟随对话历史
            messages = [
                {'role': 'user', 'content': content}
            ]

            # 调用 Dashscope 的 Generation API
            response = dashscope.Generation.call(
                self.model_name,
                messages=messages,
                result_format='message',
            )

            # 检查响应状态码
            if response.status_code == HTTPStatus.OK:
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
                print('请求失败 - Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    getattr(response, 'request_id', 'N/A'),
                    getattr(response, 'status_code', 'N/A'),
                    getattr(response, 'code', 'N/A'),
                    getattr(response, 'message', 'N/A')
                ))
                return None

        except Exception as e:
            print(f"调用模型时发生异常: {e}")
            return None
