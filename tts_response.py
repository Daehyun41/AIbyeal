import os
from pathlib import Path
from openai import OpenAI
import base64
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
from utils import generate_unique_filename, create_directory_structure

# Load API key from .env file
load_dotenv()
SECRET_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=SECRET_KEY)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def classify_image_sentiment(image_path):
    base64_image = encode_image(image_path)

    #프롬프트 엔지니어링을 통해 답변을 긍정/중립/부정으로 제한시키기.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 얼굴 표정을 분석하는 유용하고 좋은 전문가입니다. 모든 대화는 한국어로 이루어집니다."},
            {"role": "user", "content": [
                {"type": "text", "text": "다음 사진을 보고 사람이 어떤 표정을 짓고 있는지 구분해주세요. 답변은 긍정, 부정, 중립 셋 중 하나여야 합니다."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content


def speak_reward(facial_result: str, character: str, scenario: str, reaction: str, voice: str = "shimmer", base_directory: str = "gpt_tts_results"):
    text = ""
    if facial_result == '긍정':
        text = "굉장히 밝은 표정입니다. 아주 잘했어요!"
    elif facial_result == '중립':
        text = "좀 더 밝은 표정을 지어주세요!"
    elif facial_result == '부정':
        text = "우울한 표정을 지으면 안돼요. 활짝 웃어보아요!"

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )

    save_directory = create_directory_structure(base_directory, character, scenario, reaction)
    output_filename = generate_unique_filename("tts_output", "mp3")
    speech_file_path = save_directory / output_filename

    response.stream_to_file(speech_file_path)
    print(f"Audio saved as {speech_file_path}")

    # Load and play the audio file
    data, samplerate = sf.read(speech_file_path)
    sd.play(data, samplerate)
    sd.wait()  # Wait until the audio is done playing

def gpt4o_response():
    return "정말 좋은 생각이에요!"

def speak_speech(system_speech: str, character: str, scenario: str, reaction: str, voice: str = "shimmer", base_directory: str = "gpt_tts_results"):
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=system_speech
    )

    save_directory = create_directory_structure(base_directory, character, scenario, reaction)
    output_filename = generate_unique_filename("tts_output", "mp3")
    speech_file_path = save_directory / output_filename

    response.stream_to_file(speech_file_path)
    print(f"Audio saved as {speech_file_path}")

    # Load and play the audio file
    data, samplerate = sf.read(speech_file_path)
    sd.play(data, samplerate)
    sd.wait()  # Wait until the audio is done playing

# Example usage
if __name__ == "__main__":
    character = "Poly"
    scenario = "School"
    reaction = "Positive"
    
    # Example of generating and saving TTS output based on facial recognition result
    facial_result = "긍정"  # This would be dynamically determined in a real scenario
    speak_reward(facial_result, character, scenario, reaction)
    
    # Example of generating and saving TTS output based on GPT-4o's response
    system_speech = gpt4o_response()
    speak_speech(system_speech, character, scenario, reaction)
