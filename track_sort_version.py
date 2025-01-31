import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity
import utils.config_detect as config
import csv
from utils.sort import Sort
import argparse
import os

def draw_trajectories(frame, tracks, thickness=2):
    """Draws the trajectories of tracked objects on the frame using cv2.polylines."""
    for track in tracks:
        # Get the color for the track based on the track ID
        track_color = (0, 255, 255)
        # Collect all points in the track
        if len(track.centroidarr) > 1:  # Ensure there are at least two points to form a polyline
            points = np.array(track.centroidarr, dtype=np.int32).reshape((-1, 1, 2))

            # Draw the polyline on the frame
            cv2.polylines(frame, [points], isClosed=False, color=track_color, thickness=thickness)

    return frame



def load_boxes_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    boxes_data = {}
    min_frame_index = min(entry['frame_index'] for entry in data)
    for entry in data:
        frame_index = entry['frame_index'] - min_frame_index
        contours = entry['contours']
        if frame_index not in boxes_data:
            boxes_data[frame_index] = contours
        else:
            boxes_data[frame_index].extend(contours)
    return boxes_data

def apply_clahe(image, clip_limit=15.0, tile_grid_size=(1, 1)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        merged_lab = cv2.merge([l, a, b])
        result_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    else:
        result_image = clahe.apply(image)
    return result_image

def apply_sharpen(image, strength=1):
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + strength, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def get_closest_points(points, box, num_points=5):
    center_x = box['x'] + box['width'] // 2
    center_y = box['y'] + box['height'] // 2
    distances = [np.linalg.norm(np.array([new[0] - center_x, new[1] - center_y])) for new, old in points]
    sorted_indices = np.argsort(distances)
    closest_indices = sorted_indices[:num_points]
    closest_points = [points[i] for i in closest_indices if not (box['x'] <= points[i][0][0] <= box['x'] + box['width'] and box['y'] <= points[i][0][1] <= box['y'] + box['height'])]
    other_points = [points[i] for i in sorted_indices[num_points:]]
    return closest_points, other_points

def outside_calculate_draw(frame, outside_points, box):
    #cv2.rectangle(frame, (box['x'], box['y']), (box['x'] + box['width'], (box['y'] + box['height'])), (0, 255, 0), 2)
    center_x = box['x'] + box['width'] // 2
    center_y = box['y'] + box['height'] // 2
    radius = 50  
    #cv2.circle(frame, (center_x, center_y), radius, (255, 255, 0), 2)

    # if old_center is not None:
    #     cv2.circle(frame, (int(new_center[0]), int(new_center[1])), 5, (0, 255, 0), -1)
    #     cv2.line(frame, (int(old_center[0]), int(old_center[1])), (int(new_center[0]), int(new_center[1])), (0, 255, 0), 2)

    closest_points, other_points = get_closest_points(outside_points, box)
    #print(f"Closest Points: {closest_points}")
    # for new, old in closest_points:
    #     a, b = map(int, new.ravel())
    #     c, d = map(int, old.ravel())
    #     cv2.circle(frame, (a, b), 5, (255, 0, 0), -1)  # 蓝色
    #     cv2.line(frame, (c, d), (a, b), (255, 0, 0), 2)

    # for new, old in other_points:
    #     a, b = map(int, new.ravel())
    #     c, d = map(int, old.ravel())
    #     cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
    #     cv2.line(frame, (c, d), (a, b), (0, 0, 255), 2)

    #avg_inside_vector = np.array(new_center) - np.array(old_center) if old_center is not None else np.array([0, 0])
    avg_outside_vector = calculate_avg_vector(closest_points)
    #vector_diff = np.linalg.norm(avg_inside_vector - avg_outside_vector)
    #overall_movement = np.linalg.norm(avg_inside_vector) + np.linalg.norm(avg_outside_vector)
    #cv2.putText(frame, f'Vector Diff: {vector_diff:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return  avg_outside_vector


def calculate_avg_vector(points):
    vectors = [new - old for new, old in points]
    avg_vector = np.mean(vectors, axis=0) if len(vectors) else np.array([0, 0])
    return avg_vector

def plot_vectors(ax, points, vectors, frame_idx, color):
    for point, vector in zip(points, vectors):
        x, y = point
        u, v = vector[0], vector[1]
        ax.quiver(x, y, frame_idx, u, v, 0, color=color, length=5, arrow_length_ratio=0.08)
    ax.set_xlabel('X-axis (pixels)')
    ax.set_ylabel('Y-axis (pixels)')
    ax.set_zlabel('Frame Index')

def remove_smallest_vectors(points, percentage=0.2):
    if not points:
        return []

    lengths = [np.linalg.norm(new - old) for new, old in points]
    sorted_indices = np.argsort(lengths)
    num_to_remove = int(len(points) * percentage)
    largest_indices = sorted_indices[num_to_remove:]
    filtered_points = [points[i] for i in largest_indices]
    return filtered_points

def cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  
    dot_product = np.dot(v1, v2)
    return dot_product / (norm_v1 * norm_v2)

def magnitude_difference(v1, v2):
    max_magnitude = max(np.linalg.norm(v1), np.linalg.norm(v2), 1)
    return abs(np.linalg.norm(v1) - np.linalg.norm(v2)) / max_magnitude

def draw_labeled_box(frame, d, label, font_scale=0.5, thickness=2, bg_color=(252, 46, 255), text_color = (255, 255, 255)):
    x1, y1, x2, y2, track_id = d
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (252, 46, 255), 2)
    # 计算中心点坐标
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # 在中心绘制一个小圆点
    cv2.circle(frame, (center_x, center_y), radius=3, color=(255, 255, 0), thickness=-1)

    # 获取文本大小
    ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # 计算文本背景的顶部左角和底部右角
    label_bg_top_left = (x1, y1 - text_height - 10)
    label_bg_bottom_right = (x1 + text_width, y1)
    cv2.rectangle(frame, (int(label_bg_top_left[0]), int(label_bg_top_left[1])), (int(label_bg_bottom_right[0]), int(label_bg_bottom_right[1])), bg_color, cv2.FILLED)

    # 添加文本
    text_org = (int(x1), int(y1 - 5))
    cv2.putText(frame, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)


def process_video(video_path, json_path, roi_points, update_interval, frame_fast):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / video_fps / frame_fast)
    lk_params = dict(winSize=(35, 35), maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.000003))
    boxes_data = load_boxes_data(json_path)

    frame_idx = 1
    id_vectors = {}
    id_cosine_similarities = {}
    id_magnitude_diffs = {}

    ret, old_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Couldn't read video")

    if len(roi_points) > 2:
        ROI_mask = np.zeros(old_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(ROI_mask, [np.array(roi_points, dtype=np.int32)], 255)

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = apply_clahe(old_gray)
    old_gray = apply_sharpen(old_gray)
    old_gray = cv2.bitwise_and(old_gray, old_gray, mask=ROI_mask)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=200, qualityLevel=0.01, minDistance=20, useHarrisDetector=False, blockSize=3)

    features = []
    id_vectors = {}
    last_centers = {}  # Dictionary to store the last centers for each track_id

    output_path = video_path.replace('.mp4', '_output.mp4')
    frame_height, frame_width = old_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))
    sort_tracker = Sort(max_age=300, iou_threshold=0.1, min_hits=3)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=ROI_mask)
        frame_gray = apply_clahe(frame_gray)
        frame_gray = apply_sharpen(frame_gray)

        current_boxes = boxes_data.get(frame_idx, [])
        detections = []
        if current_boxes is None:
            current_boxes = []
        for current_box in current_boxes:
            x1, y1, w, h = current_box['x'], current_box['y'], current_box['width'], current_box['height']
            detections.append([x1, y1, x1 + w, y1 + h])

        if len(detections):
            tracked_dets = sort_tracker.update(np.array(detections))
            tracks =sort_tracker.getTrackers()

            draw_trajectories(frame, tracks)
            
            for d in tracked_dets:
                x1, y1, x2, y2, track_id = d
                #cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                draw_labeled_box(frame, d, f"ID: {int(track_id)}")
                new_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if track_id not in id_vectors:
                    id_vectors[track_id] = {"inside_vectors": [], "outside_vectors": []}
                    id_cosine_similarities[track_id] = []
                    id_magnitude_diffs[track_id] = []

                    last_centers[track_id] = new_center  # Initialize the center for new track_id

                # Calculate avg_inside_vector based on the last known center and the new center
                if track_id in last_centers:
                    avg_inside_vector = np.array(new_center) - np.array(last_centers[track_id])
                    id_vectors[track_id]["inside_vectors"].append((new_center[0], new_center[1], avg_inside_vector.tolist(), frame_idx))

                # Update the last known center
                last_centers[track_id] = new_center
        else:   #SORT should be updated even with no detections
            tracked_dets = sort_tracker.update()
                

        if frame_idx > 3:
            if frame_idx % update_interval == 0:
                p0 = None
                old_gray = frame_gray.copy()

            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                if p1 is not None and len(p1[st == 1]) > 0:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    if current_boxes:
                        for current_box in current_boxes:
                            new_center = (current_box['x'] + current_box['width'] // 2, current_box['y'] + current_box['height'] // 2)

                            # Collect points outside of the current bounding box.
                            outside_points = []
                            err = 5  # Error margin for point inclusion outside the box.
                            for (new, old) in zip(good_new, good_old):
                                a, b = new.ravel()
                                if not (current_box['x'] - err <= a <= current_box['x'] + current_box['width'] + err and 
                                        current_box['y'] - err <= b <= current_box['y'] + current_box['height'] + err):
                                    outside_points.append((new, old))
                            
                            # Filter out the smallest vectors to reduce noise.
                            outside_points = remove_smallest_vectors(outside_points, 0.1)
                            avg_outside_vector = outside_calculate_draw(frame, outside_points, current_box)  # Assuming a modified function to calculate average vector.

                            # Check for minimal movement in the outside vector, if negligible, treat it as zero movement.
                            if abs(avg_outside_vector[1]) < 0.1 and 0 < abs(avg_outside_vector[0]) < 0.1:
                                id_cosine_similarities[track_id].append(0)
                            else:
                                if track_id in last_centers:
                                    cos_sim = cosine_similarity(avg_inside_vector, avg_outside_vector)
                                    id_cosine_similarities[track_id].append(cos_sim)
                                    mag_diff = magnitude_difference(avg_inside_vector, avg_outside_vector)
                                    id_magnitude_diffs[track_id].append(mag_diff)

                            id_vectors[track_id]["outside_vectors"].append((new_center[0], new_center[1], avg_outside_vector.tolist(), frame_idx))

                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=20, useHarrisDetector=False, blockSize=3)
        #cv2.imshow('Frame', frame)
        out.write(frame)

        # if cv2.waitKey(delay) & 0xFF == ord('q'):
        #     break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    max_frames_id = max(id_vectors, key=lambda k: len(id_vectors[k]["inside_vectors"]))
    #print(id_vectors)
    print(f"ID with most frames: {max_frames_id}, len: {len(id_vectors[max_frames_id]['inside_vectors'])}")
    inside_vectors = id_vectors[max_frames_id]["inside_vectors"]
    outside_vectors = id_vectors[max_frames_id]["outside_vectors"]
    magnitude_diff = id_magnitude_diffs[max_frames_id]
    cosine_sim = id_cosine_similarities[max_frames_id]
    
    return inside_vectors, outside_vectors, magnitude_diff, cosine_sim


def compute_similarity(frame_cosine_similarities, frame_magnitude_diffs):
    # 计算每一帧的cosine similarity平均值和标准差
    if frame_cosine_similarities:
        mean_cosine_similarity = np.nanmean(frame_cosine_similarities)
        std_cosine_similarity = np.nanstd(frame_cosine_similarities)
        threshold = 2 * std_cosine_similarity  # 您可以根据需要调整这个阈值

        # 删除偏差超过阈值的outliers
        filtered_cosine_similarities = [cos_sim for cos_sim in frame_cosine_similarities if abs(cos_sim - mean_cosine_similarity) <= threshold]
    else:
        filtered_cosine_similarities = []

    # 计算每一帧的magnitude difference平均值和标准差
    if frame_magnitude_diffs:
        mean_magnitude_diff = np.nanmean(frame_magnitude_diffs)
        std_magnitude_diff = np.nanstd(frame_magnitude_diffs)
        threshold_magnitude = 2 * std_magnitude_diff  # 您可以根据需要调整这个阈值

        # 删除偏差超过阈值的outliers
        filtered_magnitude_diffs = [mag_diff for mag_diff in frame_magnitude_diffs if abs(mag_diff - mean_magnitude_diff) <= threshold_magnitude]
    else:
        filtered_magnitude_diffs = []

    avg_cosine_similarity = np.mean(filtered_cosine_similarities) if filtered_cosine_similarities else 0
    avg_magnitude_diff = np.mean(filtered_magnitude_diffs) if filtered_magnitude_diffs else 0
    #avg_overall_movement = np.mean(frame_overall_movements) if frame_overall_movements else 0

    # Normalize overall movement to a [0, 1] range (you may need to adjust the max value based on your data)
    #normalized_overall_movement = avg_overall_movement / (avg_overall_movement + 1)
    
    combined_similarity = 0.5 * avg_cosine_similarity + 0.5 * (1 - avg_magnitude_diff) #+ 0.2 * normalized_overall_movement
    print(f"Average Cosine Similarity: {avg_cosine_similarity:.2f}")
    print(f"Average Magnitude Difference: {avg_magnitude_diff:.2f}")
    #print(f"normalize Average Overall Movement: {normalized_overall_movement:.2f}")
    print(f"Combined Similarity: {combined_similarity:.2f}")
    return avg_cosine_similarity, avg_magnitude_diff, combined_similarity

def save_vectors_to_json(inside_vectors, outside_vectors, avg_cosine_similarity, avg_magnitude_diff, combined_similarity, result, json_path):
    data = {
        "inside_vectors": inside_vectors,
        "outside_vectors": outside_vectors,
        "avg_cosine_similarity": avg_cosine_similarity,
        "avg_magnitude_diff": avg_magnitude_diff,
        "combined_similarity": combined_similarity,
        "result": result
    }

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
        

def plot_all_vectors(inside_vectors, outside_vectors, avg_cosine_similarity, avg_magnitude_diff, combined_similarity, result, plot_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制内部向量
    for (x, y, vector, frame_idx) in inside_vectors:
        plot_vectors(ax, [(x, y)], [vector], frame_idx, 'blue')

    # 绘制外部向量
    for (x, y, vector, frame_idx) in outside_vectors:
        plot_vectors(ax, [(x, y)], [vector], frame_idx, 'red')

    # 添加平均余弦相似度、平均大小差异、组合相似度和识别结果的文本
    ax.text2D(0.3, 0.95, f"Avg Cosine Similarity: {avg_cosine_similarity:.2f}", transform=ax.transAxes)
    ax.text2D(0.3, 0.90, f"Avg Magnitude Diff: {avg_magnitude_diff:.2f}", transform=ax.transAxes)
    ax.text2D(0.3, 0.85, f"Combined Similarity: {combined_similarity:.2f}", transform=ax.transAxes)
    ax.text2D(0.3, 0.80, f"Result: {result}", transform=ax.transAxes)
    plt.savefig(plot_path, dpi=300)  # 指定文件名和分辨率
    json_path = plot_path.replace('_plot.png', '_vectors.json') 
    save_vectors_to_json(inside_vectors, outside_vectors, avg_cosine_similarity, avg_magnitude_diff, combined_similarity, result, json_path)

    plt.show()

def main():
    # test for a folder of videos
    # folder_root = 'debug/2024-07-10_629_225_654_493_big_rain/best'
    
    # # Iterate through each folder in the root directory
    # for folder_name in os.listdir(folder_root):
    #     folder_path = os.path.join(folder_root, folder_name)
        
    #     if os.path.isdir(folder_path):
    #         video_path = os.path.join(folder_path, f"{folder_name}.mp4")
            
    #         if os.path.exists(video_path):
    #             json_path = video_path.replace('.mp4', '_contours.json')
    #             plot_path = video_path.replace('.mp4', '_plot.png')
    #             #roi_points = [(3, 204), (104, 182), (253, 170), (408, 161), (557, 161), (663, 158), (802, 161), (904, 173), (983, 184), (984, 737), (3, 736), (2, 208)]
    #             roi_points = [(0, 142), (55, 126), (178, 112), (328, 106), (489, 109), (588, 111), (651, 125), (652, 491), (2, 489), (4, 142)]

    #             inside_vectors, outside_vectors, magnitude_diff, cosine_sim = process_video(video_path, json_path, roi_points, update_interval=5, frame_fast=2)

    #             avg_cosine_similarity, avg_magnitude_diff, combined_similarity = compute_similarity(cosine_sim, magnitude_diff)
    #             label = "Active" if combined_similarity < 0.5 else "passive"
    #             if len(inside_vectors) < 10:
    #                 label = "Other abnormal"
    #             plot_all_vectors(inside_vectors, outside_vectors, avg_cosine_similarity, avg_magnitude_diff, combined_similarity, label, plot_path)
    parser = argparse.ArgumentParser(description="處理異常影片資料夾")
    parser.add_argument("--anomaly_folder", type=str, help="包含異常影片的資料夾路徑")
    args = parser.parse_args()
    points = []
    with open("points.txt", "r") as f:
        for line in f:
            x, y = map(int, line.strip().split(','))
            points.append((x, y))
    file_name = os.path.basename(args.anomaly_folder)
    video_path = os.path.join(args.anomaly_folder, f"{file_name}.mp4")
    ### test for a single video
    #video_path = 'debug/0808/92040_93010/92040_93010.mp4'  #humane_log_file, bird_get, test ,348390_348492, 199364_199646
    #video_path = 'debug/0802/262194_262301.mp4'    #262194_262301 
    #video_path = 'debug/0808/545201_545368/545201_545368.mp4'    #323726_323821 #1524_1610
    #video_path = 'debug/2024-07-10_629_225_654_493_big_rain/final_new/524177_524285/524177_524285.mp4'  #525764_525904(太少case), 524177_524285
    #video_path = 'debug/2024-07-10_629_225_654_493_big_rain/final_new/464322_464436/464322_464436.mp4'  #599467_599898 (bird slow) ,414052_414152 (bird fast), 199364_199646
    #video_path =  'debug/四分溪河道監視器_hurricane/38452_38581/38452_38581.mp4'  #38596_39769, 41638_42034

    json_path = video_path.replace('.mp4', '_contours.json')
    #roi_points = [(3, 204), (104, 182), (253, 170), (408, 161), (557, 161), (663, 158), (802, 161), (904, 173), (983, 184), (984, 737), (3, 736), (2, 208)]
    #roi_points = [(0, 142), (55, 126), (178, 112), (328, 106), (489, 109), (588, 111), (651, 125), (652, 491), (2, 489), (4, 142)]
    inside_vectors, outside_vectors, magnitude_diff, cosine_sim = process_video(video_path, json_path, points, update_interval=5, frame_fast=15)

    avg_cosine_similarity, avg_magnitude_diff, combined_similarity = compute_similarity(cosine_sim, magnitude_diff)
    label = "Birds" if combined_similarity < 0.5 else "passive"
    if(len(inside_vectors) < 10):
        label = "Other abnormal"
        
    plot_path = video_path.replace('.mp4', '_plot.png')
    txt_path = video_path.replace('.mp4', '_result.txt')
    with open(txt_path, "w") as f:
        f.write(label)
    print(f"檢測結果已儲存到：{txt_path}")
    #plot_all_vectors(inside_vectors, outside_vectors, avg_cosine_similarity, avg_magnitude_diff, combined_similarity, label, plot_path)
    

if __name__ == "__main__":
    main()
