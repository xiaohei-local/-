from ultralytics import YOLO

model = YOLO(r"yolov8n-pose.pt")

model.predict(
    source=r"pose_test1.mp4",
    save=False,
    show=True,
)
