import openai
import webbrowser
import os
import urllib.request
import time

# OpenAI API 키 설정
client = openai.OpenAI(api_key="YOUR_API_KEY")

# 이미지 생성 요청
response = client.images.generate(
  model="dall-e-3",
  prompt="A Pororo-like penguin character is having fun with his friends",
  size="1024x1024",
  quality="standard",
  n=1,
)

# 생성된 이미지 URL 열기
url = response.data[0].url
webbrowser.open(url)

# 이미지 다운로드 및 저장
img_dest = "./result.jpg"

start = time.time()

# URL에서 이미지 다운로드
urllib.request.urlretrieve(url, img_dest)

end = time.time()
print(f"총 소요시간 {end-start}초")

print(f"이미지가 {img_dest}에 저장되었습니다.")