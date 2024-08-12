import openai
# import webbrowser
import urllib.request
from PIL import Image, ImageDraw, ImageFont
import os
import time

# OpenAI API 키 설정
client = openai.OpenAI(api_key="API_KEY")

# 좋아하는 캐릭터와 상황을 변수로 설정
character = "뽀로로를 닮은 펭귄 캐릭터"
scenario = "친구들과 스키를 타며 놀고 있는 상황"


# 대사를 자동 생성하기 위한 함수
def generate_dialogue(scenario, num_lines=4):
    # 대화 생성 프롬프트
    dialogue_prompt = f"Generate {num_lines} lines of dialogue for a scenario where {scenario}."

    # GPT-4 모델을 사용하여 대사 생성
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "당신은 주어진 시나리오에 맞는 대화 내용을 생성하는 유용하고 좋은 전문가입니다. 모든 대화는 한국어로 이루어집니다."},
            {"role": "user", "content": dialogue_prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )

    # 생성된 대사를 리스트로 변환
    dialogues = response.choices[0].message.content.strip().split('\n')
    return dialogues


# 대화 목록 설정
dialogues = generate_dialogue(scenario)


# 이미지 생성 및 Seed 번호 확보
def generate_base_image_with_seed(character, scenario):
    # 이미지 프롬프트 생성
    prompt = f"{character}, {scenario}"

    # 이미지 생성 시작 시간 기록
    start_time = time.time()

    # 이미지 생성 요청
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        # response_format="b64_json",
        size="1024x1024",
    )

    # 이미지 생성 완료 시간 기록
    end_time = time.time()

    # 생성된 이미지 URL과 Seed 번호

    url = response.data[0].url
    seed = response.created

    # 생성 시간 계산
    generation_time = end_time - start_time

    return url, seed, generation_time

def generate_next_image_with_seed(character, scenario, seed):
    # 이미지 프롬프트 생성
    prompt = f"{character}, {scenario}, {seed}와 같은 캐릭터로"

    # 이미지 생성 시작 시간 기록
    start_time = time.time()

    # 이미지 생성 요청
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        # response_format="b64_json",
        size="1024x1024",
    )

    # 이미지 생성 완료 시간 기록
    end_time = time.time()

    # 생성된 이미지 URL과 Seed 번호

    url = response.data[0].url

    # 생성 시간 계산
    generation_time = end_time - start_time

    return url, generation_time


# 기본 이미지를 다운로드 및 Seed 번호 확보
base_image_url, seed, generation_time = generate_base_image_with_seed(character, scenario)
url_2, generation_time2 = generate_next_image_with_seed(character, scenario, seed)
url_3, generation_time3 = generate_next_image_with_seed(character, scenario, seed)
url_4, generation_time4 = generate_next_image_with_seed(character, scenario, seed)
base_image_path = './test4_3/images/image.jpg'
image_path2 = './test4_3/images/image2.jpg'
image_path3 = './test4_3/images/image3.jpg'
image_path4 = './test4_3/images/image4.jpg'
urllib.request.urlretrieve(base_image_url, base_image_path)
urllib.request.urlretrieve(url_2, image_path2)
urllib.request.urlretrieve(url_3, image_path3)
urllib.request.urlretrieve(url_4, image_path4)

# 이미지 생성 시간 출력
print(f"기본 이미지 생성 완료: 소요 시간 {generation_time:.2f}초, 시드 번호: {seed}")
print(f"두번째 이미지 생성 완료: 소요 시간 {generation_time2:.2f}초")
print(f"세번째 이미지 생성 완료: 소요 시간 {generation_time3:.2f}초")
print(f"네번째 이미지 생성 완료: 소요 시간 {generation_time4:.2f}초")
# 말풍선을 추가하는 함수
def add_speech_bubble(image_path, dialogue, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 말풍선 텍스트 설정
    text = dialogue.strip()

    # 폰트 설정 (한글 지원 폰트 필요)
    font_path = "./font/나눔손글씨 고딕 아니고 고딩.ttf"  # 한글을 지원하는 폰트 파일 경로
    font_size = 30
    font = ImageFont.truetype(font_path, font_size)

    # 말풍선 그리기
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    bubble_padding = 10

    # 말풍선 배경 그리기
    bubble_x0, bubble_y0 = 50, 50  # 말풍선의 시작 좌표 (수정 가능)
    bubble_x1 = bubble_x0 + text_width + 2 * bubble_padding
    bubble_y1 = bubble_y0 + text_height + 2 * bubble_padding
    draw.rectangle([bubble_x0, bubble_y0, bubble_x1, bubble_y1], fill="white", outline="black")

    # 말풍선 텍스트 그리기
    text_x = bubble_x0 + bubble_padding
    text_y = bubble_y0 + bubble_padding
    draw.text((text_x, text_y), text, font=font, fill="black")

    # 수정된 이미지를 저장합니다.
    image.save(output_path)


# 출력 폴더 생성
output_dir = "./test4_3"
os.makedirs(output_dir, exist_ok=True)

# 각 대사에 대해 동일한 Seed 번호로 이미지 생성 및 말풍선 추가
for i, dialogue in enumerate(dialogues):
    # 기존 이미지에서 말풍선만 추가
    output_path = os.path.join(output_dir, "result.jpg")
    output_path2 = os.path.join(output_dir, "result_2.jpg")
    output_path3 = os.path.join(output_dir, "result_3.jpg")
    output_path4 = os.path.join(output_dir, "result_4.jpg")
    add_speech_bubble(base_image_path, dialogues[0], output_path)
    add_speech_bubble(image_path2, dialogues[1], output_path2)
    add_speech_bubble(image_path3, dialogues[2], output_path3)
    add_speech_bubble(image_path4, dialogues[3], output_path4)
    print(f"이미지 {i + 1} 저장 완료: {output_path}")

print("모든 이미지 생성 및 저장이 완료되었습니다.")