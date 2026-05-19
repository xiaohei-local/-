# main.py
import cv2
import torch
from collections import deque
from ultralytics import YOLO
from stgcn_inference import ActionRecognizer

# ========== 配置参数 ==========
VIDEO_PATH = 'pose_test1.mp4'          # 输入视频路径
TRACKER_CONFIG = 'bytetrack_custom.yaml' # ByteTrack配置文件路径
CONFIDENCE_THRESHOLD = 0.5             # 检测置信度阈值
ACTION_BUFFER_SIZE = 30                # 动作识别所需帧数（1秒30帧）
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 动作标签映射（根据你的模型调整）==========
# Kinetics-400数据集的类别映射示例
ACTION_LABELS = {
    0: "Walking",
    1: "Running",
    2: "Jumping",
    3: "Sitting",
    4: "Standing",
    5: "Falling",
    # 根据实际模型补充...
}
# ========== 初始化组件 ==========
print("正在加载YOLOv8-Pose模型...")
yolo_model = YOLO('yolov8n-pose.pt')

print("正在加载ST-GCN模型...")
action_recognizer = ActionRecognizer(device=DEVICE)

# 为每个跟踪目标维护关键点队列
pose_buffers = {}  # {track_id: deque}

# ========== 视频处理主循环 ==========
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"错误：无法打开视频 {VIDEO_PATH}")
    exit()

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLOV8检测 + ByteTrack跟踪
    results = yolo_model.track(
        frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        conf=CONFIDENCE_THRESHOLD,
        iou=0.5
    )

    # 2. 处理检测结果
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # 提取数据
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

        # 建立track_id到边界框的映射，用于可视化
        id_to_box = {tid: box for tid, box in zip(track_ids, boxes)}

        # 3. 逐目标处理关键点累积和动作识别
        for track_id, kpt in zip(track_ids, keypoints):
            # 提取关键点坐标 (x, y)，忽略置信度
            keypoints_xy = kpt[:, :2]

            # 初始化或更新队列
            if track_id not in pose_buffers:
                pose_buffers[track_id] = deque(maxlen=ACTION_BUFFER_SIZE)
            pose_buffers[track_id].append(keypoints_xy)

            # 4. 当队列满时进行ST-GCN推理
            if len(pose_buffers[track_id]) == ACTION_BUFFER_SIZE:
                # 执行动作识别
                action_id, confidence = action_recognizer.predict(
                    list(pose_buffers[track_id])
                )

                # 获取动作标签
                action_name = ACTION_LABELS.get(action_id, f"Unknown({action_id})")

                # 5. 在边界框上方绘制动作识别结果
                if track_id in id_to_box:
                    x1, y1, x2, y2 = id_to_box[track_id].astype(int)
                    text = f"ID:{track_id} | {action_name} ({confidence:.2f})"
                    cv2.putText(
                        frame, text, (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

                # 可选：滑窗预测（保留最后10帧，继续累积新帧）
                # for _ in range(10):
                #     pose_buffers[track_id].popleft()

        # 6. 绘制所有跟踪框和ID
        for tid, box in id_to_box.items():
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f'ID: {tid}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

    # 7. 显示结果
    cv2.imshow('YOLOv8 + ByteTrack + ST-GCN', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("运行结束！")