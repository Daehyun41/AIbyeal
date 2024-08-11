import os
import shutil
import time

def find_files(directories):
    audio_files = {}
    json_files = []

    for directory in directories:
        if not os.path.exists(directory):
            print(f'Directory does not exist: {directory}')
            continue

        for root, dirs, files in os.walk(directory):
            try:
                wav_files = []
                m4a_files = []
                for file in sorted(files):
                    if file.lower().endswith('.wav'):
                        wav_files.append(os.path.join(root, file))
                    elif file.lower().endswith('.m4a'):
                        m4a_files.append(os.path.join(root, file))
                    elif file.lower().endswith('.json'):
                        json_files.append(os.path.join(root, file))
                if wav_files or m4a_files:
                    audio_files[root] = {'wav': wav_files, 'm4a': m4a_files}
            except PermissionError:
                print(f'Permission denied: {root}')
                continue

    print(f"Found {len(audio_files)} directories with audios, and {len(json_files)} JSON files.")
    return audio_files, json_files

def save_files_with_structure(audio_files, save_directory, base_directory):
    for root_dir, file_types in audio_files.items():
        for file_type, files in file_types.items():
            for file_path in files:
                relative_path = os.path.relpath(file_path, start=base_directory)
                target_path = os.path.join(save_directory, relative_path)

                target_dir = os.path.dirname(target_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                shutil.copy(file_path, target_path)
                print(f"Copied {file_path} to {target_path}")

def extract_youtube_audio(youtube_url, save_directory, audio_filename="extracted_audio.wav"):
    try:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        audio_filepath = os.path.join(save_directory, audio_filename)

        command = f'yt-dlp -x --audio-format wav -o "{audio_filepath}" "{youtube_url}"'
        os.system(command)

        if os.path.exists(audio_filepath):
            print(f"Audio extracted successfully as {audio_filepath}!")
            return audio_filepath
        else:
            print("Failed to extract audio.")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
