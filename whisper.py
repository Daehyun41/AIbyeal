"""
Whisper 단계 실험)

1. AIHub의 '한국어 아동 음성 데이터'를 바탕으로 STT 인식률 확인
(link: 'https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=540')

2. YouTube의 한국어 아동 영상 샘플을 바탕으로 STT 인식률 확인
(마이크에서 바로 송출되는 경우가 아닌, 영상에서 오디오 추출의 경우 시나리오 확인)

3. 실시간 안드로이드 폰에서 영상 촬영/음성 녹음 후 STT 인식률 확인
(1과 2의 경우, 실시간과 다를 수 있어 실제 즉각적인 환경에서의 시나리오 확인)

*** 최종적으로 STT 다음 GPT-4까지 연결하는 방향까지 완료.
"""

import os
import hashlib
import openai
from dotenv import load_dotenv
from openai import OpenAI
import shutil
from pydub import AudioSegment
import asyncio
import aiohttp
import time
import zipfile
import atexit
import multiprocessing
from pydub.playback import play
from find import find_files, extract_youtube_audio, save_files_with_structure
from gpt4omini import process_with_gpt4omini_async

# load openai api key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_unique_filename(base_name, extension):
    timestamp = int(time.time())
    unique_hash = hashlib.md5(base_name.encode()).hexdigest()[:8]
    return f"{base_name}_{unique_hash}_{timestamp}.{extension}"


def split_audio(audio_file_path, chunk_length_ms=30000):
    audio = AudioSegment.from_file(audio_file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_name = f"{os.path.splitext(audio_file_path)[0]}_part{i // chunk_length_ms}_{generate_unique_filename('', 'wav').split('.')[0]}.wav"
        chunk.export(chunk_name, format="wav")
        chunks.append(chunk_name)
    return chunks


async def transcribe_audio(audio_file_path):
    transcript_text = ""
    client = OpenAI()
    temp_dir = "./temp_files"  # 명시적인 임시 디렉토리 경로 지정

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_filename = None  # temp_filename 변수를 초기화합니다.

    try:
        print(f"Starting STT for {audio_file_path}...")

        # 오디오 파일을 읽어 확장자에 관계없이 처리
        audio = AudioSegment.from_file(audio_file_path)
        start_time = time.time()

        # 명시적으로 지정된 임시 디렉토리에 파일 생성 (모든 파일을 .wav 형식으로 변환)
        temp_filename = os.path.join(temp_dir, f"{os.path.basename(audio_file_path)}.wav")
        audio.export(temp_filename, format="wav")

        # 임시 파일을 열어 STT 요청을 처리
        with open(temp_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            transcript_text = transcript

        end_time = time.time()
        print(f"STT for {audio_file_path} took {end_time - start_time:.2f} seconds.")
        print(f"STT result for {audio_file_path}:\n{transcript_text}\n")

    except PermissionError:
        print(f"Permission denied error occurred while accessing {audio_file_path}.")
    except Exception as e:
        print(f"An error occurred during STT for {audio_file_path}: {e}")
    finally:
        # 임시 파일 삭제
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                print(f"Failed to remove temp file {temp_filename}: {e}")

    return transcript_text



async def process_youtube_url(url, youtube_save_directory, youtube_stt_output_directory, gpt_results_directory):
    try:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        url_directory = os.path.join(youtube_save_directory, url_hash)
        if not os.path.exists(url_directory):
            os.makedirs(url_directory)

        audio_file_path = extract_youtube_audio(url, url_directory)
        if audio_file_path:
            audio_chunks = split_audio(audio_file_path)
            full_transcript = ""
            for chunk in audio_chunks:
                transcript = await transcribe_audio(chunk)
                full_transcript += transcript

                relative_chunk_path = os.path.relpath(chunk, start=youtube_save_directory)
                transcript_file_dir = os.path.join(youtube_stt_output_directory, os.path.dirname(relative_chunk_path))
                if not os.path.exists(transcript_file_dir):
                    os.makedirs(transcript_file_dir)

                transcript_file = os.path.join(transcript_file_dir, os.path.basename(chunk).replace('.wav', '.txt'))
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(transcript)

                gpt_response, gpt_reward = await process_with_gpt4omini_async(transcript)
                gpt_file_dir = os.path.join(gpt_results_directory, os.path.dirname(relative_chunk_path))
                if not os.path.exists(gpt_file_dir):
                    os.makedirs(gpt_file_dir)

                gpt_file = os.path.join(gpt_file_dir, os.path.basename(chunk).replace(".wav", "_gpt.txt"))
                with open(gpt_file, "w", encoding="utf-8") as f:
                    f.write(gpt_response)
                    f.write("\n")
                    f.write(gpt_reward)
                    
            return full_transcript
        else:
            print("Failed to extract audio from YouTube.")
    except Exception as e:
        print(f"An error occurred while processing the YouTube URL {url}: {e}")



async def stt_from_aihub_data(dataset_directory, stt_output_directory, gpt_results_directory):
    audio_files, _ = find_files([dataset_directory])
    save_files_with_structure(audio_files, dataset_directory, dataset_directory)
    
    for dir_name, files in audio_files.items():
        for audio_file_path in files['wav']:
            print(f"Processing file: {audio_file_path}")
            
            audio_chunks = split_audio(audio_file_path)
            full_transcript = ""
            for chunk in audio_chunks:
                transcript = await transcribe_audio(chunk)
                full_transcript += transcript

                relative_chunk_path = os.path.relpath(chunk, start=dataset_directory)
                transcript_file_dir = os.path.join(stt_output_directory, "AIHub", os.path.dirname(relative_chunk_path))
                if not os.path.exists(transcript_file_dir):
                    os.makedirs(transcript_file_dir)

                transcript_file = os.path.join(transcript_file_dir, os.path.basename(chunk).replace('.wav', '.txt'))
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(transcript)

                gpt_response, gpt_reward = await process_with_gpt4omini_async(transcript)
                gpt_file_dir = os.path.join(gpt_results_directory, "AIHub", os.path.dirname(relative_chunk_path))
                if not os.path.exists(gpt_file_dir):
                    os.makedirs(gpt_file_dir)

                gpt_file = os.path.join(gpt_file_dir, os.path.basename(chunk).replace(".wav", "_gpt.txt"))
                with open(gpt_file, "w", encoding="utf-8") as f:
                    f.write(gpt_response)
                    f.write("\n")
                    f.write(gpt_reward)


def unzip_sample_data(zip_file_path, extract_to):
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                try:
                    # 먼저 기본적으로 CP437에서 UTF-8로 디코딩 시도
                    decoded_name = member.filename.encode('cp437').decode('utf-8')
                except UnicodeDecodeError:
                    # 실패하면 EUC-KR 등의 다른 인코딩을 시도
                    try:
                        decoded_name = member.filename.encode('cp437').decode('euc-kr')
                    except UnicodeDecodeError:
                        print(f"파일 이름 디코딩 실패: {member.filename}")
                        decoded_name = member.filename  # 디코딩 실패 시 원래 이름 사용

                member.filename = decoded_name
                zip_ref.extract(member, extract_to)
            print(f"Extracted {zip_file_path} to {extract_to}.")
    else:
        print(f"Zip file {zip_file_path} does not exist.")




async def main(choice):
    data_directory = "./aihub_data"
    stt_output_directory = "./stt_results"
    youtube_save_directory = "./youtube_audio_files"
    youtube_stt_output_directory = os.path.join(stt_output_directory, "YouTube")
    aihub_stt_output_directory = os.path.join(stt_output_directory, "AIHub")
    gpt_results_directory = "./gpt_results"
    gpt_youtube_directory = os.path.join(gpt_results_directory, "YouTube")
    gpt_aihub_directory = os.path.join(gpt_results_directory, "AIHub")
    sample_zip_path = "./New_Sample.zip"

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    if not os.path.exists(youtube_stt_output_directory):
        os.makedirs(youtube_stt_output_directory)
        
    if not os.path.exists(aihub_stt_output_directory):
        os.makedirs(aihub_stt_output_directory)

    if not os.path.exists(gpt_youtube_directory):
        os.makedirs(gpt_youtube_directory)

    if not os.path.exists(gpt_aihub_directory):
        os.makedirs(gpt_aihub_directory)


    if choice == "1":
        unzip_sample_data(sample_zip_path, data_directory)
        dataset_directory = "./aihub_data/"
        await stt_from_aihub_data(dataset_directory, stt_output_directory, gpt_aihub_directory)
    elif choice == "2":
        youtube_urls = [
            'https://youtu.be/0wFag0_WcaU?si=od8VkB5r1dQhByLW',
            'https://youtu.be/SKL7PbAxveE?si=e8eeT2EepJ3mmiqu'
        ]
        async with aiohttp.ClientSession() as session:
            for url in youtube_urls:
                await process_youtube_url(url, youtube_save_directory, youtube_stt_output_directory, gpt_youtube_directory)
    else:
        print("잘못된 선택입니다. 프로그램을 종료합니다.")


def run_asyncio(choice):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main(choice))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

if __name__ == "__main__":
    print("실행할 작업을 선택하세요:")
    print("1: AIHub의 한국어 아동 음성 데이터 STT")
    print("2: 유튜브의 한국어 아동 영상 샘플 STT")
    choice = input("선택 (1 또는 2): ")

    p = multiprocessing.Process(target=run_asyncio, args=(choice,))
    p.start()
    p.join()