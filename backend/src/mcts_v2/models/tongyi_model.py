from .base_model import BaseModel
from http import HTTPStatus
import os
import dashscope


class TongyiModel(BaseModel):
    def __init__(self, model_name, instruction):
        super().__init__(model_name, instruction)
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("环境变量 'DASHSCOPE_API_KEY' 未设置。")
        dashscope.api_key = api_key

    def generate_response(self, conversation):
        try:
            # 将 instruction 作为系统提示
            messages = [
                {'role': 'system', 'content': self.instruction}
            ] + conversation

            # 调用 Dashscope 的 Generation API
            response = dashscope.Generation.call(
                self.model_name,
                messages=messages,
                result_format='message',
            )

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
