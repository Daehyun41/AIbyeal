import openai
import webbrowser
import urllib.request
import time

# OpenAI API 키 설정
openai.api_key = "YOUR_API_KEY"

# 좋아하는 캐릭터와 상황을 변수로 설정
character = "뽀로로를 닮은 펭귄 캐릭터"
scenario = "친구들과 스키를 타며 놀고 있는 상황"

# 대사를 자동 생성하기 위한 함수
def generate_dialogue(scenario, num_lines=4):
    # 대화 생성 프롬프트
    dialogue_prompt = f"Generate {num_lines} lines of dialogue for a scenario where {scenario}."

    # GPT-4 모델을 사용하여 대사 생성
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": dialogue_prompt}
        ],
        max_tokens=150,
        n=1,
        temperature=0.7,
    )

    # 생성된 대사를 리스트로 변환
    dialogues = response['choices'][0]['message']['content'].strip().split('\n')
    return dialogues

# 대화 목록 설정
dialogues = generate_dialogue(scenario)

# 이미지 생성 및 저장
for i in range(len(dialogues)):
    # 각 이미지의 프롬프트 생성
    prompt = f"{character}, {scenario}, 말풍선에 '{dialogues[i].strip()}'"

    # 이미지 생성 요청
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )

    # 생성된 이미지 URL
    url = response['data'][0]['url']

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