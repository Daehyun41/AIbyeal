import os
import sys

# Suppress TensorFlow informational, warning, and error logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
import openai
import asyncio
import multiprocessing
import time
from dotenv import load_dotenv
from whisper import transcribe_audio  # whisper.py에서 가져옴
from gpt4omini_recognition import recognition_process, recognition2_process  # gpt4omini_recognition.py에서 가져옴
from gpt4omini_whisper import process_with_gpt4omini_async as whisper_process  # gpt4omini_whisper.py에서 가져옴
import sounddevice as sd
import soundfile as sf
import cv2
from PIL import Image
from utils import generate_unique_filename, unzip_image_set
from tts_response import speak_reward, speak_speech  # tts_response.py에서 가져옴

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Suppress all exceptions from showing up
def silent_excepthook(exc_type, exc_value, exc_traceback):
    pass

sys.excepthook = silent_excepthook

async def make_personalization(chosen):
    try:
        characters = ['Heartsping', 'Pinkpong', 'Poly', 'Pororo']
        character_root_path = os.path.join('image_set', 'characters', chosen)

        if chosen in characters:
            scenarios = {0: 'Public_Transportation', 1: 'Restaurant', 2: 'School'}
            reactions = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}

            chosen_scenario = random.randint(0, 2)
            chosen_reaction = random.randint(0, 2)

            directory = os.path.join(character_root_path, scenarios[chosen_scenario], reactions[chosen_reaction])

            valid_images = lambda x: x.lower().endswith(('.png', '.jpg', '.jpeg'))

            if os.path.exists(directory):
                images = sorted([os.path.join(directory, img) for img in os.listdir(directory) if valid_images(img)])
                if len(images) >= 4:
                    print(f"Chosen directory is: {directory}")
                    return images[:4], chosen, scenarios[chosen_scenario], reactions[chosen_reaction]
                else:
                    print(f"Not enough images in {directory}. Finding other directories...")
            else:
                print(f"Directory does not exist: {directory}")

            all_images = []
            for scenario in scenarios.values():
                for reaction in reactions.values():
                    dir_path = os.path.join(character_root_path, scenario, reaction)
                    if os.path.exists(dir_path):
                        images = sorted([os.path.join(dir_path, img) for img in os.listdir(dir_path) if valid_images(img)])
                        if len(images) >= 4:
                            return images[:4], chosen, scenario, reaction
                        all_images.extend(images)

            if len(all_images) >= 4:
                return sorted(random.sample(all_images, 4)), chosen, scenario, reaction
            else:
                print("Not enough images found across all directories.")
                return [], None, None, None
        else:
            print("Invalid character chosen.")
            return [], None, None, None
    except Exception as e:
        print(f"An error occurred in make_personalization: {e}")
        return [], None, None, None

def capture_look(character, scenario, reaction, capture_files_directory="capture_files"):
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Create directory structure based on character, scenario, and reaction
            save_directory = os.path.join(capture_files_directory, character, scenario, reaction)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            fps = 1  # Frames per second (1 frame per second for simplicity)
            frame_count = 10  # Total frames to capture
            interval = int(1000 / fps)  # Interval in milliseconds

            captured_frames = 0
            while captured_frames < frame_count:
                ret, img = cap.read()
                if ret:
                    # Save the frame as an image file
                    img_filename = generate_unique_filename(f"capture_{captured_frames + 1}", "jpg")
                    img_filepath = os.path.join(save_directory, img_filename)
                    cv2.imwrite(img_filepath, img)

                    # Display the image on the window
                    cv2.imshow('Capturing Look', img)

                    captured_frames += 1
                    print(f"Captured frame {captured_frames} as {img_filepath}")

                    if cv2.waitKey(interval) != -1:
                        break
                else:
                    print('Failed to capture image!')
                    break

            # Close the window after capturing
            cv2.destroyAllWindows()
        else:
            print("Can't open camera!")

        cap.release()

        return save_directory
    except Exception as e:
        print(f"An error occurred in capture_look: {e}")
        return None

async def record_audio(duration=10, samplerate=44100, channels=2, audio_save_directory="./image_net_audio", image_path=None):
    try:
        print("Recording audio automatically for {} seconds...".format(duration))

        # Ensure the save directory exists
        if not os.path.exists(audio_save_directory):
            os.makedirs(audio_save_directory)

        # Display the last image before recording
        if image_path:
            image = Image.open(image_path)
            image.show()

        # Start recording
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
        sd.wait()  # Wait until the recording is finished

        # Generate a unique filename
        temp_audio_file_name = generate_unique_filename("audio", "wav")
        temp_audio_file_path = os.path.join(audio_save_directory, temp_audio_file_name)

        # Save the recorded audio to the specified directory
        sf.write(temp_audio_file_path, recording, samplerate)

        print(f"Audio recorded and saved as {temp_audio_file_path}")

        # Close the image after recording is done
        if image_path:
            image.close()

        return temp_audio_file_path
    except Exception as e:
        print(f"An error occurred during recording: {e}")
        return None

async def main(chosen):
    try:
        image_set_path = "./image_set.zip"
        unzip_image_set(image_set_path, ".")
        
        gpt_recognition_results_directory = "./gpt_recognition_results"
        gpt_whisper_results_directory = "./gpt_whisper_results"
        
        images_directory, character, scenario, reaction = await make_personalization(chosen)
        
        if not images_directory:
            print("No images found. Exiting.")
            return

        # Create subdirectory structure based on the scenario and reaction
        result_subdir = os.path.join(gpt_recognition_results_directory, character, scenario, reaction)
        if not os.path.exists(result_subdir):
            os.makedirs(result_subdir)

        whisper_result_subdir = os.path.join(gpt_whisper_results_directory, character, scenario, reaction)
        if not os.path.exists(whisper_result_subdir):
            os.makedirs(whisper_result_subdir)

        # Show the last image from the original directory before capturing user's look
        print("Showing the last image from images_directory.")
        last_image_in_directory = images_directory[-1]
        Image.open(last_image_in_directory).show()

        # Wait for 5 seconds before starting the capture
        print("You have 5 seconds to prepare...")
        time.sleep(5)
        print("Starting to capture your expression...")

        # 사용자의 표정 녹화 촬영 및 이미지 분할 저장
        capture_directory = capture_look(character, scenario, reaction)
        
        if capture_directory is None:
            print("Error during image capture.")
            return
        
        # ###################################################
        # # 캡처된 이미지를 기반으로 감정 인식 및 GPT-4oMini 분석
        # capture_images = sorted([os.path.join(capture_directory, img) for img in os.listdir(capture_directory) if img.lower().endswith(('.jpg', '.jpeg', '.png'))])
        # expectation_results = [reaction] * len(capture_images)  # 예상 결과를 모든 이미지에 대해 동일하게 설정
        # results = await recognition_process(capture_images)
        # _, final_reward = await recognition2_process(results, expectation_results)        
        # gpt_recognition_file = os.path.join(result_subdir, f"{generate_unique_filename('recognition_result', 'txt')}")
        # with open(gpt_recognition_file, "w", encoding="utf-8") as f:
        #     f.write(final_reward)
        
        # # GPT-4oMini 분석을 토대로 TTS 진행
        # speak_reward(final_reward, character, scenario, reaction, voice="shimmer", base_directory='gpt_tts_results')
        
        # 마지막 이미지를 기반으로 오디오 녹음
        print("Prepare to record audio for the final image.")     
        audio_file_path = await record_audio(duration=10, image_path=last_image_in_directory)  # 10초 동안 자동으로 녹음
        
        if audio_file_path is None:
            print("Audio recording failed.")
            return
        
        # 녹음된 오디오 파일을 Whisper와 GPT-4oMini로 처리
        transcript = await transcribe_audio(audio_file_path)
        
        combined_input = f"Scenario: {scenario}, Reaction: {reaction}, Situation: {scenario}, Transcript: {transcript}"
        
        # GPT-4oMini Whisper 분석
        gpt_response, gpt_reward = await whisper_process(combined_input)
        
        # GPT-4oMini 결과에 관한 TTS 진행 및 저장
        speak_speech(gpt_reward, character, scenario, reaction)
        
        # GPT-4oMini 결과에 관한 파일 저장
        gpt_file_dir = os.path.join(whisper_result_subdir, f"{generate_unique_filename('whisper_result', 'txt')}")
        with open(gpt_file_dir, "w", encoding="utf-8") as f:
            f.write(gpt_response)
            f.write("\n")
            f.write(gpt_reward)
    except Exception as e:
        print(f"An error occurred: {e}")

def run_asyncio(choice):
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main(choice))
    except Exception as e:
        print(f"An error occurred in asyncio loop: {e}")
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as shutdown_error:
            print(f"An error occurred during loop shutdown: {shutdown_error}")
        finally:
            loop.close()

if __name__ == "__main__":
    print("캐릭터를 선택하세요:")
    chosen = input("선택 ('Heartsping', 'Pinkpong', 'Poly' 또는 'Pororo'): ")

    p = multiprocessing.Process(target=run_asyncio, args=(chosen,))
    p.start()
    print("Process started")
    p.join()
    print("Process finished")
