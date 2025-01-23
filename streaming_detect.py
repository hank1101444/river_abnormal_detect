import cv2
import os
import time
import utils.module_detect as md
import utils.config_detect as config  
def main():
    youtube_url = 'https://www.youtube.com/watch?v=MIAmzZAU6MQ'
    YT = md.YTStreamProcessor(youtube_url)
    stream_url = YT.get_video_stream_url(youtube_url)
    print(f"Stream URL: {stream_url}")
    # use current Date to name
    video_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    video_name = "test"
    config.image_save_path = f'abnormal/{video_name}'
    config.video_save_path = f'debug/{video_name}'
    config.txt_path = f'abnormal_log_{video_name}.txt'  
    frame_fast = 2

    with open(config.txt_path, "w") as file:
        file.write("")
        
    #video_processor = md.VideoProcessor(x=932, y=337, width=986, height=741)  
    video_processor = md.VideoProcessor(x=0, y=0, width=699, height=523)  

    try:
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Original video FPS: {video_fps}")
        delay = int(1000 / video_fps / frame_fast)
        #delay = 0  # 如果不需要調試

        ret, first_frame = cap.read()
        if not ret:
            print("Cannot read the video.")
            cap.release()
            return

        first_frame = first_frame[video_processor.y:video_processor.y+video_processor.height,
                                video_processor.x:video_processor.x+video_processor.width]
        mask = video_processor.get_roi(first_frame)
        if mask is None:
            print("ROI not defined.")
            return
        
        frame_count = 0
        
        # 調試參數
        start_frame = 0
        #end_frame = 531112
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame

        last_time = time.time()
        while True:
            # if frame_count > end_frame:
            #     print("end_frame", end_frame)
            #     break
            ret, frame = cap.read()
            if not ret:
                print("Reached end of the video or cannot fetch the frame.")
                break
            detect, frame = video_processor.detect_abnormalities(frame, frame_count, mask)
            if detect:
                print(f"Abnormal activity detected and saved at frame {frame_count}")


            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            cv2.imshow("Video", frame)
            frame_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        video_processor.cleanup()

if __name__ == '__main__':
    main()
