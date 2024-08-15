import time
import hashlib
import os
import sys
import zipfile
from pathlib import Path
import cv2


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


def unzip_image_set(zip_file_path, extract_to):
    try:
        if os.path.exists(zip_file_path):
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extracted {zip_file_path} to {extract_to}.")
        else:
            print(f"Zip file {zip_file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred while extracting {zip_file_path}: {e}")
        sys.exit(1)


def generate_unique_filename(base_name, extension):
    timestamp = int(time.time())
    unique_hash = hashlib.md5(base_name.encode()).hexdigest()[:8]
    return f"{base_name}_{unique_hash}_{timestamp}.{extension}"

def create_directory_structure(base_directory, character, scenario, reaction):
    directory = Path(base_directory) / character / scenario / reaction
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def capture_look():
    cap = cv2.VideoCapture(0)
    if cap.isOpened:
        file_path = './rec.avi'
        fps = 1 
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')            # 인코딩 포맷
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (int(width), int (height))                   # 프레임 크기
        
        out = cv2.VideoWriter(file_path, fourcc, fps, size) 
        while True:
            ret, img = cap.read()
            if ret:
                cv2.imshow('camera-recording', img)
                out.write(img)                             # 화면 저장
                if cv2.waitKey(int(1000/fps)) != -1:
                    break
            else:
                print('no file!')
                break
        out.release()                                       # 종료

    else:
        print("Can`t open camera!")

    cap.release()
    cv2.destroyAllWindows()