from http import HTTPStatus
import dashscope
dashscope.api_key="sk-d11a354da5f54a85a353072f24afb6e7"


def call_with_messages():
    messages = [
                {'role': 'user', 'content': '''
**Task:**

Evaluate the performance of a legal language model in a judicial consultation dialogue. Carefully read the conversation between the user and the assistant (infer), and assess the assistant's response based on the criteria of accuracy, completeness, relevance, and effectiveness.

**Instructions:**

- Assign a score from 0 to 5 for each criterion using the provided scoring guidelines.
- Provide specific explanations for each score, using professional legal terminology and analytical language.
- Output the results strictly in the specified JSON format for automated processing.

**Scoring Criteria:**

- **Accuracy (0-5):**
  - **5:** All legal documents, regulations, and terms mentioned are accurate.
  - **4:** ≥80% are accurate.
  - **3:** ≥60% and <80% are accurate.
  - **2:** ≥40% and <60% are accurate.
  - **1:** ≥20% and <40% are accurate.
  - **0:** Mostly incorrect or fabricated.

- **Completeness (0-5):**
  - **5:** Answered all types of legal questions posed by the user.
  - **4:** Answered ≥80% but not all types.
  - **3:** Answered ≥60% and <80% types.
  - **2:** Answered ≥40% and <60% types.
  - **1:** Answered ≥20% and <40% types.
  - **0:** Answered almost none.

- **Relevance (0-5):**
  - **5:** The case details and recommended similar cases fully reflect the user's query.
  - **4:** Reflects ≥80% but not all relevance.
  - **3:** Reflects ≥60% and <80% relevance.
  - **2:** Reflects ≥40% and <60% relevance.
  - **1:** Reflects ≥20% and <40% relevance.
  - **0:** Barely reflects relevance.

- **Effectiveness (0-5):**
  - **5:** Suggestions help resolve case-related issues, answer questions, and are actionable.
  - **4:** Helps resolve issues but answers are unclear; actionability is moderate.
  - **3:** Indirectly helps; answers are unclear; somewhat actionable.
  - **2:** Indirectly helps with few issues; answers are unclear; limited actionability.
  - **1:** Indirectly helps with few issues; answers are unclear; not actionable.
  - **0:** Does not help; does not answer the questions.

**Output Format:**

Provide the evaluation in the following JSON format:

```json
{
  "model_id": "",  // Identifier of the evaluated model
  "accuracy": {
    "score": 0,  // Score (0-5)
    "explanation": ""  // Explanation for the score
  },
  "completeness": {
    "score": 0,
    "explanation": ""
  },
  "relevance": {
    "score": 0,
    "explanation": ""
  },
  "effectiveness": {
    "score": 0,
    "explanation": ""
  },
  "average_score": 0  // Average of the four scores
}
```

**Notes:**

- Do not add any additional explanations or comments; only provide the JSON output.
- Maintain a neutral and objective tone; avoid subjective or emotional language.
- Do not omit any criterion; each must have a score and an explanation.

**Content:**

Evaluate the assistant's response (infer) in the following conversation:

```json
{
    "conversation": [
        {
            "role": "user",
            "content": "我有两个孩子，没有结婚证，但是我跟人家走了拿了结婚证会犯法坐牢吗？"
        },
        {
            "role": "assistant",
            "content": "根据《民法典》的规定，结婚应当男女双方完全自愿，任何一方不得强迫对方，也禁止个人或组织干涉。如果你与对方结婚的情况是完全自愿的，符合民法典的规定，那么你不会因为结婚而犯法坐牢。\n\n但需要注意的是，未办理结婚登记的，应该及时补办登记。因为完成结婚登记，才能确立婚姻关系，获得法律的保护。如果你和对方没有完成结婚登记，你们在法律上没有建立起婚姻关系，可能会在涉及到财产分割、子女抚养等方面产生很多问题。因此，建议你们及时到婚姻登记机关进行结婚登记，获得法律的认可和保障。\n\n除此之外，如果你有其他方面的问题或疑虑，欢迎进一步说明，我可以帮助你进行更加准确的法律分析。"
        },
        {
            "role": "user",
            "content": "噢，明白了，需要补办结婚登记，那这个需要哪些手续呀？"
        }
    ],
    "label": {
        "role": "assistant",
        "content": "根据《民法典》的规定，要求结婚的男女双方应当亲自到婚姻登记机关申请结婚登记。具体手续包括：\n\n1. 男女双方携带本人的身份证或户口簿、未婚证明等证明材料前往婚姻登记机关；\n2. 填写结婚登记申请表，并签署结婚登记申请书；\n3. 提供双方固定住所、职业、联系方式等相关信息；\n4. 双方自愿声明结婚意愿，进行个人信息确认和登记确认。\n\n具体手续可能因地区而异，建议你到当地的民政局或婚姻登记机关咨询具体的办理流程。\n\n祝你好运！"
    },
    "infer": {
        "role": "assistant",
        "content": "根据《民法典》的规定，要求结婚的男女双方应当亲自到婚姻登记机关申请结婚登记。具体手续包括：\n\n1. 男女双方携带本人的身份证或户口簿、未婚证明等证明材料前往婚姻登记机关；\n2. 填写结婚登记申请表，并签署结婚登记申请书；\n3. 提供双方固定住所、职业、联系方式等相关信息；\n4. 双方自愿声明结婚意愿，进行个人信息确认和登记确认。\n\n具体手续可能因地区而异，建议你到当地的民政局或婚姻登记机关咨询具体的办理流程。\n\n祝你好运！"  
    },
    "id": 0
}
```
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