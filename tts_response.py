import os
from pathlib import Path
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
openai.api_key = SECRET_KEY

def speak(text, voice="shimmer", output_file="shimmer.mp3"):
    """Generate speech from text and save as an MP3 file."""
    speech_file_path = Path(__file__).parent / output_file

    response = openai.Audio.create(
        model="tts-1",
        voice=voice,
        input=text
    )

    with open(speech_file_path, "wb") as audio_file:
        audio_file.write(response['audio'])

    print(f"Audio saved as {output_file}")

if __name__ == "__main__":
    text_to_speak = "안녕하세요! 반갑습니다. 지시를 따라주세요!"
    speak(text_to_speak)
