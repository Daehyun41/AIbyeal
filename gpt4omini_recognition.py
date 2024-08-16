import os
import sys
import openai
from dotenv import load_dotenv
import aiohttp
from fer import FER  # Facial Emotion Recognition
import cv2
import asyncio

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Suppress TensorFlow informational, warning, and error logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize FER model
emotion_detector = FER()

async def recognition_process(image_paths):
    try:
        results = []
        processed_images = set()  # 중복 이미지 처리 방지

        for image_path in image_paths:
            if image_path in processed_images:
                continue  # 이미 처리된 이미지 건너뛰기

            processed_images.add(image_path)
            try:
                print(f"Processing image: {image_path}")
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Image at path {image_path} could not be loaded. Skipping.")
                    continue

                # Ensure the image has three channels
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif img.ndim != 3 or img.shape[2] != 3:
                    print(f"Invalid image format for {image_path}. Skipping.")
                    continue

                emotion, score = emotion_detector.top_emotion(img)
                
                if emotion:
                    results.append(emotion)
                else:
                    print(f"No emotion detected for image: {image_path}. Defaulting to 'neutral'.")
                    results.append("neutral")

            except Exception as img_ex:
                print(f"Exception occurred while processing image {image_path}: {img_ex}. Skipping.")
                continue

        return results

    except Exception as e:
        print(f"Exception occurred in recognition_process: {e}")
        return []

async def extract_emotion_classification(gpt_result):
    classifications = ["positive", "neutral", "negative"]
    gpt_result_lower = gpt_result.lower()

    for classification in classifications:
        if classification in gpt_result_lower:
            return classification

    return None

async def recognition2_process(results, expectation_results):
    try:
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json",
        }

        success_count = 0
        final_results = []
        rewards = []

        emotion_mapping_prompt = """
        감지된 감정은 다음과 같이 분류됩니다:
        - Positive: happy, surprise
        - Neutral: neutral
        - Negative: sad, fear, disgust, angry
        """

        for idx, result in enumerate(results):
            if not result:
                print(f"Skipping entry {idx} because it is empty or None.")
                continue

            expected = expectation_results[idx].lower()
            actual = result.lower()

            print(f"Expected: {expected}, Actual: {actual}")

            prompt = f"{emotion_mapping_prompt} 감지된 감정: {result}. 주어진 감정을 Positive, Neutral, Negative 중 하나로 분류해주세요."

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "당신은 인간 상호 작용의 맥락에서 얼굴 감정을 해석하는 전문가입니다."},
                    {"role": "user", "content": prompt.strip()}
                ],
                "max_tokens": 100,
                "temperature": 0.5,
            }

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            if 'choices' in response_json and len(response_json['choices']) > 0:
                                gpt_result = response_json['choices'][0]['message']['content'].strip()
                                print(f"GPT-4o Mini Explanation: {gpt_result}")
                            else:
                                print("No valid response in choices.")
                                continue
                        else:
                            print(f"Failed with status code: {response.status}")
                            continue
                except Exception as api_ex:
                    print(f"Exception occurred during API call: {api_ex}")
                    continue

            emotion_classification = await extract_emotion_classification(gpt_result)
            if emotion_classification:
                print(emotion_classification)
                if expected == emotion_classification:
                    success_count += 1
                    rewards.append(f"상황에 적절한 표정입니다. 아주 잘했어요! {gpt_result}")
                else:
                    rewards.append(f"상황에 적절한 표정은 {expected}이랍니다! 다시 해볼까요? {gpt_result}")
                final_results.append(result)
            else:
                print("Failed to extract emotion classification from GPT response.")

        print(f"Success Counts: {success_count}")
        if success_count >= 3:
            final_reward = "잘했어요! 여러 장의 사진에서 기대한 표정을 지었어요."
        else:
            final_reward = "몇몇 사진에서 표정이 기대와 달랐어요. 다음에는 더 잘할 수 있을 거예요!"

        print(f"Final reward based on captured images: {final_reward}")
        return final_results, final_reward
    except Exception as e:
        print(f"Exception occurred in recognition2_process: {e}")
        return [], ""

async def main_process():
    try:
        # Replace with the correct image paths
        image_paths = [
            "capture_files/Poly/Restaurant/Positive/capture_1_18c967d6_1723771414.jpg",
            "capture_files/Poly/Restaurant/Positive/capture_2_91b44eac_1723771415.jpg",
            "capture_files/Poly/Restaurant/Positive/capture_3_93910d1a_1723771416.jpg",
            "capture_files/Poly/Restaurant/Positive/capture_4_3f23f8bb_1723771417.jpg",
            "capture_files/Poly/Restaurant/Positive/capture_5_944bcb74_1723771418.jpg",
            "capture_files/Poly/Restaurant/Positive/capture_6_7ae095ba_1723771419.jpg",
            "capture_files/Poly/Restaurant/Positive/capture_7_ac44ab57_1723771420.jpg",
            "capture_files/Poly/Restaurant/Positive/capture_8_ba9d1dec_1723771422.jpg",
            "capture_files/Poly/Restaurant/Positive/capture_9_3483daa8_1723771423.jpg",
            "capture_files/Poly/Restaurant/Positive/capture_10_6a07ac11_1723771424.jpg"
        ]

        results = await recognition_process(image_paths)
        expectation_results = ["positive"] * 10  # Expected results for all 10 images

        final_results, final_reward = await recognition2_process(results, expectation_results)

    except Exception as e:
        print(f"An error occurred in main_process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main_process())
