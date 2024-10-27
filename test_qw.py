from http import HTTPStatus
import dashscope
dashscope.api_key=""

instruction="""
You are an advanced legal assistant. Now, you are provided with several previous conversations between a user and an assistant. Based on your professional legal knowledge, please respond to the user's last statement in the conversation history. Your response should align with the user's actual needs and provide detailed advice that complies with legal regulations.

###Conversation: 
{{question_text}}

###Response:
"""
def call_with_messages():
    messages = [
                {'role': 'user', 'content': '''
'''}]
    response = dashscope.Generation.call(
        "farui-plus",
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


if __name__ == '__main__':
    call_with_messages()