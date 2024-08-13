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
scenario = "친구들과 스키를 타며 즐겁게 놀고 있는 상황"


# 대사를 자동 생성하기 위한 함수
def generate_dialogue(scenario, num_lines=4):
    # 대화 생성 프롬프트
    dialogue_prompt = f"{scenario}에 대한 아동이 이해할 수 있는 완전한 문장의 {num_lines}개의 대사를 생성하세요. 각 대사는 줄바꿈으로 구분되어야 하고, 완성된 문장으로 서로 주고받는 4개의 문장이 꼭 생성되어야 합니다. dialogue list의 인자는 꼭 4개여야 합니다."

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
def generate_image(character, scenario, dialogue, seed=None):
    # 이미지 프롬프트 생성
    if seed is None:
        prompt = f"{character}가 '{dialogue}'라고 말하며, {scenario}에 있는 모습, 이미지에 텍스트, 말풍선은 절대 넣지마."
    else:
        prompt = f" 시드 번호 {seed} 사용해서 똑같은 캐릭터로 만들어줘, {character}가 '{dialogue}'라고 말하며, {scenario}에 있는 모습, 이미지에 텍스트, 말풍선은 절대 넣지마. "

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
    if seed is None:
        seed = response.created

    # 생성 시간 계산
    generation_time = end_time - start_time

    return url, seed, generation_time

# def add_speech_bubble(image_path, dialogue, output_path):
#     image = Image.open(image_path)
#     draw = ImageDraw.Draw(image)
#
#     # 말풍선 텍스트 설정
#     text = dialogue.strip()
#
#     # 폰트 설정 (한글 지원 폰트 필요)
#     font_path = "./font/나눔손글씨 고딕 아니고 고딩.ttf"  # 한글을 지원하는 폰트 파일 경로
#     font_size = 30
#     font = ImageFont.truetype(font_path, font_size)
#
#     # 말풍선 그리기
#     text_bbox = draw.textbbox((0, 0), text, font=font)
#     text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
#     bubble_padding = 10
#
#     # 말풍선 배경 그리기
#     bubble_x0, bubble_y0 = 50, 50  # 말풍선의 시작 좌표 (수정 가능)
#     bubble_x1 = bubble_x0 + text_width + 2 * bubble_padding
#     bubble_y1 = bubble_y0 + text_height + 2 * bubble_padding
#     draw.rectangle([bubble_x0, bubble_y0, bubble_x1, bubble_y1], fill="white", outline="black")
#
#     # 말풍선 텍스트 그리기
#     text_x = bubble_x0 + bubble_padding
#     text_y = bubble_y0 + bubble_padding
#     draw.text((text_x, text_y), text, font=font, fill="black")
#
#     # 수정된 이미지를 저장합니다.
#     image.save(output_path)


# 출력 폴더 생성
output_dir = "./test4_4"
os.makedirs(output_dir, exist_ok=True)

# 각 대사에 대해 동일한 Seed 번호로 이미지 생성 및 말풍선 추가
seed = None
for i, dialogue in enumerate(dialogues):
    # 첫 번째 이미지를 생성하여 시드 번호를 확보
    if i == 0:
        image_url, seed, generation_time = generate_image(character, scenario, dialogue)
    else:
        image_url, _, generation_time = generate_image(character, scenario, dialogue, seed)

    # 이미지 다운로드 및 말풍선 추가
    image_path = os.path.join(output_dir, f"image_{i+1}.jpg")
    urllib.request.urlretrieve(image_url, image_path)
    # output_path = os.path.join(output_dir, f"result_{i+1}.jpg")
    # add_speech_bubble(image_path, dialogue, output_path)

    # print(f"이미지 {i + 1} 저장 완료: {output_path}, 소요 시간 {generation_time:.2f}초")
    print(f"이미지 {i + 1} 저장 완료: {image_path}, 소요 시간 {generation_time:.2f}초")
print("모든 이미지 생성 및 저장이 완료되었습니다.")