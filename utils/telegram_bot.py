import os
import time
import mimetypes
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import config_detect as config

TOKEN = config.tg_token  
CHAT_ID = config.tg_chat_id          

def send_photo(file_path):
    url = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'
    files = {'photo': open(file_path, 'rb')}
    data = {'chat_id': CHAT_ID}
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        print(f"成功上傳圖片 {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"上傳圖片失敗：{e}")

def send_video(file_path):
    url = f'https://api.telegram.org/bot{TOKEN}/sendVideo'
    files = {'video': open(file_path, 'rb')}
    data = {
        'chat_id': CHAT_ID,
        'supports_streaming': True  # 明確指出支援串流，避免轉為 GIF
    }
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        print(f"成功上傳影片 {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"上傳影片失敗：{e}")


def send_text(file_path):
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
    with open(file_path, 'r') as file:
        content = file.read()
    data = {'chat_id': CHAT_ID, 'text': content}
    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        print(f"成功上傳文字檔案內容 {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"上傳文字檔案內容失敗：{e}")

def process_file(file_path):
    file_name = os.path.basename(file_path)
    if file_name.endswith('output.mp4'):
        time.sleep(5)  
        send_video(file_path)
    elif file_name.endswith('result.txt'):
        send_text(file_path)
    else:
        print(f"忽略文件 {file_path}")

class MediaFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            print(f"检测到新文件：{file_path}")
            process_file(file_path)

if __name__ == "__main__":
    path = '/Users/hank/Documents/GitHub/River_abnormal_detect/debug/test' 
    event_handler = MediaFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print(f"開始監控目錄：{path}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
