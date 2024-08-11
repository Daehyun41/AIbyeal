import openai
from dotenv import load_dotenv
import os
import time
import aiohttp

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

async def process_with_gpt4_async(text):
    try:
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "당신은 유용한 도우미입니다."},
                {"role": "user", "content": f"다음 텍스트를 요약해 주세요: {text}"}
            ],
            "max_tokens": 150,
            "temperature": 0.5,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as response:
                if response.status == 200:
                    response_json = await response.json()
                    return response_json['choices'][0]['message']['content']
                else:
                    return f"GPT-4 API 요청 중 오류가 발생했습니다: {response.status}"
    except Exception as e:
        return f"예외가 발생했습니다: {e}"
