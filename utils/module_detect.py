import subprocess
import cv2
import os
import numpy as np
import time
import json
from datetime import datetime
import utils.config_detect as config
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import boxcox
import logging
from datetime import datetime, timedelta
import seaborn as sns
import utils.line_bot as lb

class VideoProcessor:
    def __init__(self, x, y, width, height, start_time_str="2024-07-10 9:20:44", fps=24):
                # 初始化日志记录
        logging.basicConfig(filename='video_processor.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.first_detection = True
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area_thresh = 75
        self.var_thresh = 25
        self.contours_pool = []
        self.abnormal_frames = [] 
        self.abnormal_frames_user = [] 
        self.abnormal_counter = 0  
        self.has_recorded_zero = False
        self.decay_delay_counter = 0  
        self.initial_delay = 3     
        self.max_abnormal_counter = 0
        #雜訊太多時要提高history
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False) #150
        self.frames_num = 0
        self.mean_area = 0
        self.manual_area_thresh_adjustment = False  
        self.high_foreground_count = 0  # 用來計數超過50前景點的frame數量
        self.mod = False 
        self.fps = fps  # 每秒帧数
        
        # 初始化时间
        self.start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        
    def get_current_time(self, frame_count):
        time_offset = timedelta(seconds=frame_count / self.fps)
        return self.start_time + time_offset


    def cleanup(self):
        """清理資源，釋放記憶體，確保不會有資源洩露"""
        self.contours_pool.clear()
        self.abnormal_frames.clear()
        self.abnormal_frames_user.clear()
        self.backSub = None

    def update_contours_pool(self, contours, upper_bound):
        self.contours_pool.append([cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) < upper_bound])
        if len(self.contours_pool) > 2000:
            self.contours_pool.pop(0)

    def upbound_thresh(self, k=4): 
        try:
            areas = np.concatenate(self.contours_pool)
        except:
            print("error")
            return None
        if len(areas) < 100:
            print("short")
            return None
        
        return np.mean(areas) + k * np.std(areas)
    
    
    def draw_area_distribution(self, areas):
        std_dev = np.std(areas)
        mean = np.mean(areas)
        sample_count = len(areas)  # 获取 areas 的数量
        k = 3
        
        # 计算偏度和峰度
        skewness = stats.skew(areas, bias=False)
        kurtosis = stats.kurtosis(areas, bias=False, fisher=True)

        # 绘制 areas 的直方图
        plt.figure(figsize=(10, 6))
        #plt.hist(areas, bins=50, color='blue', alpha=0.7, kde=True)
        plt.figure(figsize=(10, 6))
        sns.histplot(areas, bins=50, kde=True, color='blue', alpha=0.7)
    
        # 绘制均值和标准差线
        plt.axvline(mean, color='green', linestyle='--', label=f'Mean Area: {mean:.2f}')
        plt.axvline(mean + std_dev, color='orange', linestyle='--', label=f'Mean + 1 Std: {mean + std_dev:.2f}')
        plt.axvline(mean + 2 * std_dev, color='red', linestyle='--', label=f'Mean + 2 Std: {mean + 2 * std_dev:.2f}')
        #plt.axvline(self.area_thresh, color='purple', linestyle='--', label=f'Area Threshold (Mean + {k} Std): {self.area_thresh:.2f}')
        
        # 标注标准差的位置
        plt.text(mean + std_dev, plt.ylim()[1] * 0.8, '1 Std Dev', color='orange', rotation=90, verticalalignment='center')
        plt.text(mean + 2 * std_dev, plt.ylim()[1] * 0.8, '2 Std Dev', color='red', rotation=90, verticalalignment='center')

        # 添加偏度和峰度信息
        plt.text(0.02, 0.95, f'Skewness: {skewness:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))
        plt.text(0.02, 0.90, f'Kurtosis: {kurtosis:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

        plt.title(f'Distribution of Areas (Sample Size: {sample_count})')
        plt.xlabel('Area')
        plt.ylabel('Frequency')
        plt.legend()

        # 保存图像到文件，文件名包含当前的帧数
        os.makedirs('distri', exist_ok=True)
        filename = f'distri/areas_distribution_{self.frames_num}.png'
        plt.savefig(filename)
        plt.close()

        print(f"Plot saved as {filename}")


    import numpy as np

    def update_current_area_threshold(self, k=3): 
        try:
            areas = np.concatenate(self.contours_pool)
        except Exception as e:
            print("Error concatenating contours_pool:", e)
            return None

        if len(areas) == 0:
            print("No data in contours_pool to calculate threshold.")
            return None

        percent = np.percentile(areas, 25)
        filtered_areas = areas[areas > percent]
        areas = filtered_areas

        if len(filtered_areas) == 0:
            print("No data left after filtering.")
            return None

        min_area = np.min(filtered_areas)
        adjusted_areas = filtered_areas - min_area

        log_areas = np.log1p(adjusted_areas)  

        mean_log_area = np.mean(log_areas)
        std_log_area = np.std(log_areas)

        original_mean_area = np.exp(mean_log_area) - 1 + min_area
        original_std_area = (np.exp(std_log_area) - 1) * np.exp(mean_log_area)

        self.area_thresh = original_mean_area + k * original_std_area

        if self.frames_num is not None and self.frames_num % 1000 == 0:
            print(f"Current Area Threshold: {self.area_thresh:.2f}, varThreshold: {self.var_thresh:.2f}")
            #self.draw_area_distribution(log_areas)


            

    def process_detected_contours(self, sorted_contours, frame, frame_count):
        contours_info = []
        detect = False
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area > self.area_thresh:
                detect = True
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                contour_dict = {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "center_x": center_x,
                    "center_y": center_y,
                    "area": area
                }
                contours_info.append(contour_dict)

                #self.save_cropped_frame(frame, x, y, w, h, area, frame_count)

        return contours_info, detect

    def save_cropped_frame(self, frame, x, y, w, h, area, frame_count):
        if config.image_save_path is None:
            raise ValueError("The image_save_path is None. Please provide a valid path.")

        os.makedirs(config.image_save_path, exist_ok=True)
        side_length = max(w, h)
        center_x, center_y = x + w // 2, y + h // 2
        start_x = max(0, center_x - side_length // 2)
        start_y = max(0, center_y - side_length // 2)
        end_x = min(frame.shape[1], start_x + side_length)
        end_y = min(frame.shape[0], start_y + side_length)
        cropped_square = frame[start_y:end_y, start_x:end_x]
        resized_square = cv2.resize(cropped_square, (224, 224))

        frame_filename = f"{config.image_save_path}/abnormal_{frame_count}_{len(self.abnormal_frames)}_{area}.jpg"
        cv2.imwrite(frame_filename, resized_square)

    def handle_abnormal_detection(self, contours_info, frame, frame_count):

        self.abnormal_counter += 1
        self.abnormal_frames.append((frame_count, frame.copy(), contours_info))
        self.max_abnormal_counter = max(self.max_abnormal_counter, self.abnormal_counter)
        self.update_detection_log(frame_count, self.abnormal_counter)
        if self.abnormal_counter == 0 and self.has_recorded_zero:
            self.has_recorded_zero = False
            
        # max_abnormal_counter >= 500 reset
        if len(self.abnormal_frames) >= 500:
            self.save_video_and_line_bot()
            current_time = self.get_current_time(frame_count)
            logging.info(f"[{current_time}] Max abnormal counter reached 500, saving video and resetting state.")
            self.abnormal_counter = 0   #### test
            self.reset_abnormal_states()

    def decay_abnormal_counter(self, frame_count):
        if self.decay_delay_counter < self.initial_delay:
            decay_step = 1
        else:
            decay_step = 2 ** ((self.decay_delay_counter - self.initial_delay) // 5) # 5 frames decay speed 2 times
        self.abnormal_counter -= decay_step
        if self.abnormal_counter < 0:
            self.abnormal_counter = 0
        self.decay_delay_counter += 1
        self.update_detection_log(frame_count, self.abnormal_counter)

    def handle_no_detection(self, clean_frame, user_frame, frame_count):
        if self.abnormal_counter > 0:
            self.decay_abnormal_counter(frame_count)
            self.abnormal_frames.append((frame_count, clean_frame.copy(), None))
            self.abnormal_frames_user.append(user_frame.copy())
            if self.abnormal_counter == 0:
                if self.max_abnormal_counter >= 40:
                    self.save_video_and_line_bot()
                self.reset_abnormal_states()
        else:
            if self.has_recorded_zero and self.max_abnormal_counter >= 20:
                self.save_video_and_line_bot()
            self.reset_abnormal_states()
            
    def save_video_and_line_bot(self):
        user_video_path = self.save_abnormal_video(self.abnormal_frames)
        self.save_user_abnormal_video(user_video_path, self.abnormal_frames_user)
        folder = f'{config.video_save_path}/{config.start_frame}_{config.end_frame}'
        txt_path = self.track_sort(folder)
        #lb.send_line_message(config.user_id, txt_path)
        
        
        
    def reset_abnormal_states(self):
        ## 超過500的異常並沒有reset abnormal_counter
        self.abnormal_frames.clear()
        self.abnormal_frames_user.clear()
        self.max_abnormal_counter = 0
        self.has_recorded_zero = True  
        self.decay_delay_counter = 0

    def rec_area_on_frame(self, frame, sorted_contours):
        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area > self.area_thresh:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {area}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                self.abnormal_frames_user.append(frame.copy())

    def detect_abnormalities(self, frame, frame_count, roi_mask):
        self.frames_num += 1
        frame = frame[self.y:self.y+self.height, self.x:self.x+self.width]

        if(self.frames_num < 100):
            foreground_mask = self.backSub.apply(frame)
            return False, frame
        # very chaos: history = 30, varTrheshold = 200
        var_thresh_min = 170#80  
        var_thresh_max = 250
        area_thresh_min = 75
        area_thresh_max = 450

        # 線性插值計算 varThreshold
        #print("mean_area: ", self.mean_area)
        self.var_thresh = var_thresh_min + (self.area_thresh - area_thresh_min) * (var_thresh_max - var_thresh_min) / (area_thresh_max - area_thresh_min)
        self.var_thresh = max(min(self.var_thresh, var_thresh_max), var_thresh_min)  # 保證 varThreshold 在範圍內
        self.backSub.setVarThreshold(self.var_thresh)
        current_time = self.get_current_time(frame_count)
        # if self.area_thresh > 200:
        #     if not self.mod:  # 如果当前是第一次超过 250
        #         self.mod = True
        #         self.backSub.setHistory(30)
        #         logging.info(f"[{current_time}] Area threshold exceeded 250, setting history to 30")
        # elif self.area_thresh <= 200:
        #     if self.mod:  # 如果当前是第一次低于或等于 250
        #         self.mod = False
        #         self.backSub.setHistory(60)
        #         logging.info(f"[{current_time}] Area threshold dropped to 250 or below, setting history to 60")

        
        
       
        foreground_mask = self.backSub.apply(frame)
        
        
        foreground_mask = cv2.bitwise_and(foreground_mask, foreground_mask, mask=roi_mask)
        
        blurred_mask = cv2.GaussianBlur(foreground_mask, (5, 7), 0) #57
        #('foreground_mask', blurred_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #55
        dilated_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_OPEN, kernel)
        if(frame_count== 39362):
            #39362
            cv2.imwrite("foreground_mask.jpg", foreground_mask)
            cv2.imwrite("frame.jpg", frame)
            cv2.imwrite("blurred_mask.jpg", blurred_mask)
            cv2.imwrite("dilated_mask.jpg", dilated_mask)
        # 计算前景中的非零像素数量
        fore_ground_num = np.sum(blurred_mask > 0)

        # 如果前景中没有检测到东西（即前景点数很少），逐步减少 area_thresh
        if fore_ground_num < 20:
            self.area_thresh = max(self.area_thresh - 20, 75)
            self.manual_area_thresh_adjustment = True  # 進入手動調整模式
            #self.high_foreground_count = 0  # 重置計數器
            #print(f"No foreground detected, reducing area_thresh to {self.area_thresh}")
        else:
            self.manual_area_thresh_adjustment = False  # 退出手動調整模式
            self.high_foreground_count = 0  # 重置計數器
            #print("Exiting manual adjustment mode after 500 frames with sufficient foreground points.")

            self.backSub.setVarThreshold(self.var_thresh)
        
    
        #cleaned_mask = cv2.erode(foreground_mask, kernel, iterations=1)
    
        # dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # dilated_mask = cv2.dilate(cleaned_mask, dilation_kernel, iterations=1)
        
        ####debug####
        #cv2.imshow('dilated_mask', dilated_mask)
        # detect every none zero pixel
        
        blur_contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sorted_contours = sorted(dilated_contours, key=cv2.contourArea, reverse=True)

        detected = False
        
        if not self.manual_area_thresh_adjustment:
            self.update_current_area_threshold()  # 只有在不處於手動調整模式時更新
        
        roi_area = np.sum(roi_mask) / 255
        upper_bound = roi_area * 0.5
        self.update_contours_pool(blur_contours, upper_bound) # use blur to update
        
        clean_frame = frame.copy() 
        
         # 添加显示当前时间的代码
        # current_time_str = self.get_current_time(frame_count).strftime("%Y-%m-%d %H:%M:%S")
        # cv2.putText(frame, current_time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        
        text = f'Frame: {frame_count}, Area Threshold: {self.area_thresh:.2f}, Var Threshold: {self.var_thresh:.2f}'
        text_position = (int(frame.shape[1] * 0.1), int(frame.shape[0] * 0.95))  # 设置文本位置为图像的10%宽度，95%高度
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if sorted_contours:
            contours_info, detected = self.process_detected_contours(sorted_contours, clean_frame, frame_count)
        if detected:
            self.handle_abnormal_detection(contours_info, clean_frame, frame_count) #use clean_frame to save
            self.rec_area_on_frame(frame, sorted_contours) #append user frame
            self.decay_delay_counter = 0
        else:
            self.handle_no_detection(clean_frame, frame, frame_count)

        return detected, frame

    def update_detection_log(self, frame_count, abnormal_counter):
        current_time = self.get_current_time(frame_count).strftime("%Y-%m-%d %H:%M:%S")
        with open(config.txt_path, "a") as file:
            file.write(f"{current_time},{frame_count},{abnormal_counter}, {self.area_thresh}, {self.var_thresh}\n")

    def save_user_abnormal_video(self, save_path, frames):
        if not frames:
            return
        save_path = save_path.split('.')[0] + "_user.mp4"
        print("Saving user video...:", save_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, fourcc, 24.0, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

    def save_abnormal_video(self, frames):
        if not frames:
            return
        print("Saving abnormal video...")
        
        start_frame, tmp, _ = frames[0]
        end_frame, _, _ = frames[-1]
        print(f"Start frame: {start_frame}, End frame: {end_frame}")
        config.start_frame = start_frame
        config.end_frame = end_frame
        os.makedirs(config.video_save_path, exist_ok=True)
        folder = f'{config.video_save_path}/{config.start_frame}_{config.end_frame}'
        os.makedirs(folder, exist_ok=True)
        output_filename = f'{folder}/{start_frame}_{end_frame}.mp4'
        json_filename = f'{folder}/{start_frame}_{end_frame}_contours.json'

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_filename, fourcc, 24.0, (tmp.shape[1], tmp.shape[0]))
            
        all_contours_info = []
        for frame_index, frame, contours_info in frames:
            video_writer.write(frame)
            all_contours_info.append({"frame_index": frame_index, "contours": contours_info})

        video_writer.release()
        with open(json_filename, 'w') as f:
            json.dump(all_contours_info, f, indent=4)
        self.save_frames(frames, folder)
        return output_filename

    def save_frames(self, frames, save_path):
        largest_frame = None
        max_area = -1

        for frame_index, frame, contours_info in frames:
            if contours_info:
                current_max_area = max(contour['area'] for contour in contours_info)
                if current_max_area > max_area:
                    max_area = current_max_area
                    largest_frame = (frame_index, frame)

        if largest_frame:
            frame_index, frame = largest_frame
            frame_filename = f"{save_path}/top_frame_{frame_index}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {frame_index} as {frame_filename}")
        else:
            print("No frame was saved. No contours found.")


    def draw_polygon(self, event, x, y, flags, param):
        global points, roi_defined
        img = param

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(img, points[-2], points[-1], (255, 0, 0), 2)
        
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(points) > 2:
                cv2.line(img, points[-1], points[0], (0, 255, 0), 2)
                roi_defined = True


    def get_roi(self, frame):
        global points, roi_defined
        points = []
        roi_defined = False
        cv2.namedWindow('Define ROI')
        cv2.setMouseCallback('Define ROI', self.draw_polygon, frame)

        while not roi_defined:
            cv2.imshow('Define ROI', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()

        if len(points) > 2:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
            with open("points.txt", "w") as f:
                for point in points:
                    f.write(f"{point[0]}, {point[1]}\n")
            roi_area = np.sum(mask) / 255
            #area_threshold = 25
            #print("area_threshold: ", area_threshold)
            return mask
        return None
    
    def track_sort(self, folder_path):
        if os.path.exists(folder_path):
            print(f"running track_sort.py -> {folder_path}")
            subprocess.run(["python", "track_sort_version.py", "--anomaly_folder", folder_path])
            print("Anomaly detected and saved.")
            file_name = os.path.basename(folder_path)
            txt_path = os.path.join(folder_path, f"{file_name}_result.txt")
            return txt_path
        return None
    

class YTStreamProcessor:
    def __init__(self, stream_url):
        self.stream_url = stream_url

    def get_video_stream_url(self, youtube_url, format_code='232'):
        """使用 yt-dlp 獲取指定解析度的 YouTube 影片 URL。

        Args:
        youtube_url (str): YouTube 影片的 URL。
        format_code (str): 指定的格式代码或解析度。默认为 '137' (1080p)。

        Returns:
        str: 影片的 URL。
        """
        cmd_get_url = [
            'yt-dlp',
            '-f', format_code,  # 使用指定的格式代码
            '-g',  # 獲取影片流的 URL
            self.stream_url
        ]
        result = subprocess.run(cmd_get_url, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception(f"yt-dlp error: {result.stderr.decode('utf-8')}")
        video_url = result.stdout.decode('utf-8').strip()
        return video_url
    
    def get_live_url(self, youtube_url):
        """使用 yt-dlp 獲取 YouTube 直播流的 URL。

        Args:
        youtube_url (str): YouTube 直播的 URL。

        Returns:
        str: 直播流的 URL。
        """
        cmd_get_url = [
            'yt-dlp',
            '-g',  # get streaming URL
            youtube_url
        ]
        result = subprocess.run(cmd_get_url, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception(f"yt-dlp error: {result.stderr.decode('utf-8')}")
        stream_url = result.stdout.decode('utf-8').strip()
        return stream_url