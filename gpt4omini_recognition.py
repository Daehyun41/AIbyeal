import os

# Suppress TensorFlow informational, warning, and error logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set the environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import openai
from dotenv import load_dotenv
import aiohttp
from fer import FER  # Facial Emotion Recognition
import cv2
import asyncio
import re

# Rest of your imports
import tensorflow as tf

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize FER model
emotion_detector = FER()

async def recognition_process(image_paths):
    """
    This function uses a pre-trained FER model to analyze emotions in images.
    """
    try:
        results = []  # List to store detected emotions
        unique_image_paths = list(set(image_paths))  # Remove duplicate image paths

        for image_path in unique_image_paths:
            img = cv2.imread(image_path)
            emotion, score = emotion_detector.top_emotion(img)
            if emotion:  # Ensure the emotion is not None
                results.append(emotion)
            else:
                print(f"No emotion detected for image: {image_path}. Defaulting to 'neutral'.")
                results.append("neutral")  # Default to neutral if no emotion is detected

        return results

    except Exception as e:
        print(f"Exception occurred: {e}")
        return []

async def extract_emotion_classification(gpt_result):
    """
    Extract the emotion classification (Positive, Neutral, Negative) from the GPT-4o Mini response.
    """
    classifications = ["positive", "neutral", "negative"]
    
    # Lowercase the GPT result to make the search case insensitive
    gpt_result_lower = gpt_result.lower()
    
    # Check each classification and see if it appears in the gpt_result
    for classification in classifications:
        if classification in gpt_result_lower:
            return classification

    # If no classification is found, return None or a default value
    return None

async def recognition2_process(results, expectation_results):
    try:
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json",
        }

        success_count = 0  # Count of how many results matched expectations
        final_results = []  # Final list of results
        rewards = []  # List of rewards/feedback

        # Define the emotion mapping as part of the system instructions
        emotion_mapping_prompt = """
        감지된 감정은 다음과 같이 분류됩니다:
        - Positive: happy, surprise
        - Neutral: neutral
        - Negative: sad, fear, disgust, angry
        """

        unique_results = list(set(results))  # Remove duplicates in results

        for idx, result in enumerate(unique_results):
            if not result:
                print(f"Skipping entry {idx} because it is empty or None.")
                continue
            
            expected = expectation_results[idx].lower()
            actual = result.lower()

            print(f"Expected: {expected}, Actual: {actual}")

            # Ensure the result string is not empty or too short
            if not actual or len(actual) < 1:
                print(f"Skipping entry with insufficient length: {actual}")
                continue

            # Use GPT-4oMini to map the raw emotion to Positive, Neutral, or Negative
            prompt = f"{emotion_mapping_prompt} 감지된 감정: {result}. 주어진 감정을 Positive, Neutral, Negative 중 하나로 분류해주세요."

            if len(prompt.strip()) < 5:  # Validate the prompt length before sending it to the API
                print(f"Prompt too short, skipping: {prompt}")
                continue

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "당신은 인간 상호 작용의 맥락에서 얼굴 감정을 해석하는 전문가입니다."},
                    {"role": "user", "content": prompt.strip()}  # Strip whitespace for safety
                ],
                "max_tokens": 100,
                "temperature": 0.5,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        if 'choices' in response_json and len(response_json['choices']) > 0:
                            gpt_result = response_json['choices'][0]['message']['content'].strip()
                            print(f"GPT-4o Mini Explanation: {gpt_result}")
                        else:
                            gpt_result = "No valid response received."
                            print("No valid response in choices.")
                    else:
                        gpt_result = f"Failed with status code: {response.status}"
                        print(gpt_result)
                        continue  # Skip to the next result if the request failed

            # Extract the emotion classification using the updated extraction method
            emotion_classification = await extract_emotion_classification(gpt_result)
            if emotion_classification:
                print(emotion_classification)
                if expected == emotion_classification:  # Case-insensitive comparison
                    success_count += 1
                    rewards.append(f"상황에 적절한 표정입니다. 아주 잘했어요! {gpt_result}")
                else:
                    rewards.append(f"상황에 적절한 표정은 {expected}이랍니다! 다시 해볼까요? {gpt_result}")
                final_results.append(result)
            else:
                print("Failed to extract emotion classification from GPT response.")

        # Final reward based on the number of successful matches
        print(f"Success Counts: {success_count}")
        if success_count >= 3:
            final_reward = "잘했어요! 여러 장의 사진에서 기대한 표정을 지었어요."
        else:
            final_reward = "몇몇 사진에서 표정이 기대와 달랐어요. 다음에는 더 잘할 수 있을 거예요!"

        print(f"Final reward based on captured images: {final_reward}")
        return final_results, final_reward
    except Exception as e:
        print(f"Exception occurred: {e}")
        return [], ""


async def main_process():
    try:
        # Example usage; replace with your actual process
        image_paths = ["path_to_image1.jpg", "path_to_image2.jpg"]  # Example paths
        results = await recognition_process(image_paths)
        expectation_results = ["positive", "neutral"]  # Example expected results

        final_results, final_reward = await recognition2_process(results, expectation_results)

        print(f"Final reward based on captured images: {final_reward}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Start the main process
if __name__ == "__main__":
    asyncio.run(main_process())
