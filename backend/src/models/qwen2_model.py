import requests
import json
from http import HTTPStatus

class ChatModelClient:
    def __init__(self, api_url, api_key=None):
        """
        初始化 ChatModelClient。

        :param api_url: 模型 API 的 URL。
        :param api_key: 可选的 API 密钥，如果需要身份验证。
        """
        self.api_url = api_url
        self.api_key = api_key

    def generate_response(self, instruction, conversation):
        """
        调用 FaruiPlus 模型生成回复。

        :param instruction: 评估指令，包含占位符的字符串。
        :param conversation: 对话历史，列表形式，每个元素为字典，包含 'role' 和 'content'。
        :return: 模型生成的回复文本，或 None（如果调用失败）。
        """
        headers = {
            'Content-Type': 'application/json',
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        # 填充指令中的占位符，假设占位符为 {conversation}
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        system_content = instruction.format(conversation=conversation_text)

        # 构建消息数组，将 instruction 作为 'system' 消息，后面跟随对话历史
        messages = [
            {"role": "system", "content": system_content}
        ] + conversation

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.7,
            "top_p": 1.0,
            "n": 1,
            "max_tokens": 1024,
            "do_sample": True,
            "stream": False
        }

        try:
            response = requests.post(self.api_url + "/v1/chat/completions", headers=headers, data=json.dumps(payload))
            
            # 检查响应状态码
            if response.status_code == HTTPStatus.OK:
                response_data = response.json()
                if ('choices' in response_data and
                    len(response_data['choices']) > 0 and
                    'message' in response_data['choices'][0]):
                    
                    return response_data['choices'][0]['message']['content'].strip()
                else:
                    print(f"响应结构不符合预期: {response_data}")
                    return None
            else:
                print(f"请求失败 - 状态码: {response.status_code}, 响应: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"请求发生异常: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    api_url = "http://localhost:8000"  # 模型 API 的本地 URL
    model_name = "farui-plus"
    api_key = None  # 如果需要身份验证，可以在此处填写 API 密钥

    client = ChatModelClient(api_url, api_key)

    # 示例对话历史
    conversation = [
        {"role": "user", "content": "你好，今天的天气怎么样？"},
        {"role": "assistant", "content": "你好！今天的天气很晴朗，非常适合外出散步。"},
        {"role": "user", "content": "太好了，那我该穿什么？"}
    ]

    # 示例指令
    instruction = "请根据以下对话生成回复：\n{conversation}"

    # 调用模型生成回复
    response = client.generate_response(instruction, conversation)
    
    if response:
        print("模型回复: ", response)
    else:
        print("未能生成回复。")
