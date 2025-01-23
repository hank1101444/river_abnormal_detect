from linebot import LineBotApi

video_save_path = None
txt_path = None
image_save_path = None
start_frame = None
end_frame = None    
### init line_bot_api
line_bot_api = LineBotApi("++44xjAqn8haqrc/HDcUcy6KWpW1ihsFoyZ1WGuXpJ6x1gb11cV3U88ye/msw8SYnfW/8WqBhQlxMGd87NGz5L2rGFiXkD09lFChMYivH8oH/5SpBJZ0YyWXIxRCS1zHyTHc3Z6l2SmFbjVdJgpDgAdB04t89/1O/w1cDnyilFU=")  # 將 "YOUR_CHANNEL_ACCESS_TOKEN" 替換為你的 Channel Access Token
user_id = "U5afff4c1c5dbc6cdb488c9e7bc5eb585"  # Line ID
tg_token = "7455340197:AAGj4x9GYpdEcJWlAu7_G6VeXMbEK6qAlp0" #tg bot token
tg_chat_id = "-1002485671954" #tg chat id
# https://api.telegram.org/bot7978169585:AAEPwKpapon1sAbOXuDvY7MSa3_HagJGdqU/getUpdates