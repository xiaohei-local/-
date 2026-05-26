import cv2

from ultralytics import YOLO

# 1. 加载模型（姿态估计模型）
model = YOLO("yolov8n-pose.pt")

# 2. 打开视频
cap = cv2.VideoCapture("pose_test1.mp4")

# 可选：获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 3. 指定 ByteTrack 配置文件（YAML）
#    该文件可以放在当前目录，或者使用 Ultralytics 自带的默认配置
#    这里使用自定义配置文件 'bytetrack.yaml'（见下方配置内容）
tracker_config = "bytetrack.yaml"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. 执行跟踪（自动融合检测+跟踪，返回带有 track_id 的结果）
    results = model.track(frame, persist=True, tracker=tracker_config)

    # 5. 可视化（自动画出边框、ID、关键点）
    annotated_frame = results[0].plot()  # 一次性绘制所有信息
    cv2.imshow("YOLOv8 Pose + ByteTrack (Unified Config)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
