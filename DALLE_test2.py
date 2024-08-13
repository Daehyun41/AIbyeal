import openai
import webbrowser
import urllib.request
import time

# OpenAI API 키 설정
client = openai.OpenAI(api_key="YOUR_API_KEY")

# 좋아하는 캐릭터와 상황을 변수로 설정
character = "펭귄 캐릭터"
scenario = "친구들과 스키를 타며 놀고 있는 상황"

# 대화 목록 설정
dialogues = [
    "와, 이곳은 정말 멋져!",
    "내가 스키를 타는 모습을 봐!",
    "우리 다음엔 어디로 갈까?",
    "이렇게 즐거운 날은 처음이야!"
]

# 이미지 생성 및 저장
for i in range(4):
    # 각 이미지의 프롬프트 생성
    prompt = f"{character}, {scenario}, 말풍선에 '{dialogues[i]}'"

    # 이미지 생성 요청
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    # 생성된 이미지 URL
    url = response.data[0].url

    # 웹 브라우저로 이미지 열기
    webbrowser.open(url)

    # 이미지 다운로드 및 저장
    img_dest = f"./test2/result_{i+1}.jpg"

    start = time.time()

    # URL에서 이미지 다운로드
    urllib.request.urlretrieve(url, img_dest)

    end = time.time()
    print(f"이미지 {i+1} 저장 완료: {img_dest} (총 소요시간 {end-start}초)")

print("모든 이미지 생성 및 저장이 완료되었습니다.")