import openai
from dotenv import load_dotenv
import os
import time
import aiohttp

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

async def process_with_gpt4omini_async(text):
    try:
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "당신은 유용한 도우미입니다."},
                {"role": "user", "content": f"{text}가 긍정인지, 부정인지, 중립인지 셋 중 하나를 선택해 단어로만 답해주세요."}
            ],
            "max_tokens": 50,
            "temperature": 0.5,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as response:
                if response.status == 200:
                    response_json = await response.json()
                    result = response_json['choices'][0]['message']['content'].strip()

                    # 먼저 GPT-4oMini의 응답을 출력
                    print(f"GPT-4o Mini 응답: {result}")

                    # 그 후에 조건에 따라 추가 메시지를 출력
                    if result == '긍정':
                        reward = '상황에 적절한 답변입니다. 아주 잘했어요!'
                        print(reward)
                    elif result == '중립':
                        reward = '상황에 보다 적극적인 답변을 해주세요!'
                        print(reward)
                    elif result == '부정':
                        reward = '좀더 긍정적인 답변을 해보아요!'
                    else:
                        print('알 수 없는 응답입니다:', result)

                    return result, reward
                else:
                    return f"GPT-4o Mini API 요청 중 오류가 발생했습니다: {response.status}"
    except Exception as e:
        return f"예외가 발생했습니다: {e}"