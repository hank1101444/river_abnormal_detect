from linebot import LineBotApi

video_save_path = None
txt_path = None
image_save_path = None
start_frame = None
end_frame = None    
### init line_bot_api
line_bot_api = LineBotApi("YOUR_LINE_BOT_API")  # 將 "YOUR_CHANNEL_ACCESS_TOKEN" 替換為你的 Channel Access Token
user_id = "YOUR_LINE_ID"  # Line ID
tg_token = "YOUR_TG_BOT_TOKEN" #tg bot token
tg_chat_id = "YOUR_TG_CHAT_ID" #tg chat id
# https://api.telegram.org/bot{tg_token}/getUpdates
