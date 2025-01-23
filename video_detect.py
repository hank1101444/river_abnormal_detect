import cv2
import os
import time
import utils.module_detect as md
import utils.config_detect as config  
import utils.line_bot as lb

def main():
    # rain 200, 122
    video_path = "videos/test.avi"#"downloads/2024-07-10_629_225_654_493_big_rain.mkv"## #2024-07-10_629_225_654_493_big_rain, 2024-07-18 23-42-08_humane.mkv
    frame_fast = 20
    
    video_name = os.path.basename(video_path).split('.')[0]
    config.image_save_path = f'abnormal/{video_name}'
    config.video_save_path = f'debug/{video_name}'
    config.txt_path = f'abnormal_log_{video_name}.txt'  
    with open(config.txt_path, "w") as file:
        file.write("")
    
    # for fps in range(28560, 28570, 1): 
    #     fps/=1000
    #     video_processor = md.VideoProcessor(x=626, y=227, width=654, height=493, fps=fps)  # 使用 VideoProcessor 類
    #     #video_processor = md.VideoProcessor(x=932, y=337, width=986, height=741)  
    #     print(f"fps: {fps} :", video_processor.get_current_time(530980)) # 43:59
    #video_processor = md.VideoProcessor(x=626, y=227, width=654, height=493, fps=28.566)  # 使用 VideoProcessor 類
    video_processor = md.VideoProcessor(x=932, y=337, width=986, height=741)  
    cap = cv2.VideoCapture(video_path)
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
    start_frame = 2000
    end_frame = 39366
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
