from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolov8n-pose.pt")
    model.train(
        data=r"dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        cache="ram",
        workers=8,
        save=True,
        patience=20,
    )
