# River Abnormal Detect

This project is designed to detect abnormalities in river environments by analyzing video streams. It uses advanced techniques such as adaptive background subtraction, flow analysis, and object detection to classify detected objects and events. 

Demo website: https://huggingface.co/spaces/hank710535/River_Abnormal_Web

## Prerequisites

1. Install [conda](https://docs.conda.io/).
2. Use the `requirements.txt` file to set up the environment.

## Environment Setup

```bash
conda create --name river_abnormal_detect --file requirements.txt
conda activate river_abnormal_detect
```

## Modes of Operation

### 1. Offline Testing
- Run the `video_detect.py` script:
  ```bash
  python video_detect.py
  ```
- Modify the `video_path` variable in `video_detect.py` to the path of the video file you want to test.

### 2. Online Testing
- Run the `streaming_detect.py` script:
  ```bash
  python streaming_detect.py
  ```
- Modify the `youtube_url` variable in `streaming_detect.py` to include the URL of the video stream.

## Real-Time Upload of Predictions and Video Captures

There are two options for uploading predictions and captured videos in real time:

### Option 1: Telegram Bot
1. Update the `tg_token`, `tg_chat_id` field in `utils/config_detect.py` with your Telegram bot's API key.
2. When running detection, you must also execute the `utils/telegram_bot.py` script:
   ```bash
   python utils/telegram_bot.py
   ```

### Option 2: LINE Bot
1. Update the `LineBotApi("")` & `user_id` field in `utils/config_detect.py` with your LINE bot's API key.
