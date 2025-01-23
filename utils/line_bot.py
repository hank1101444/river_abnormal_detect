from linebot.v3.messaging import MessagingApi
from linebot.models import TextSendMessage
import os
import utils.config_detect as config

# 初始化 MessagingApi
#line_bot_api = MessagingApi(config.line_bot_api.access_token)  # 設定 Access Token

def send_line_message(user_id, txt_path):
    # 讀取檢測結果
    if os.path.exists(txt_path):
        with open(txt_path, "r") as file:
            result_text = file.read()
    else:
        result_text = "檢測結果不存在。"

    # 發送訊息到 Line Bot
    try:
        message = TextSendMessage(text=result_text)
        config.line_bot_api.push_message(user_id, messages=[message])  # 使用新方法
        print("檢測結果已傳送到 Line Bot")
    except Exception as e:
        print(f"無法發送訊息：{e}")

if __name__ == "__main__":
    txt_path = "../test.txt"
    send_line_message(config.user_id, txt_path)
