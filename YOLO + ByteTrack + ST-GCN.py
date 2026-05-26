"""
自动扶梯行人动作识别核心管线
YOLOv8-Pose + ByteTrack + ST-GCN（规则+深度学习混合方案）.

用法：
    python "YOLO + ByteTrack + ST-GCN.py"                         # 默认视频 pose_test1.mp4
    python "YOLO + ByteTrack + ST-GCN.py" --video xxx.mp4         # 指定视频
    python "YOLO + ByteTrack + ST-GCN.py" --camera                # 使用摄像头
    python "YOLO + ByteTrack + ST-GCN.py" --direction down        # 扶梯下行
"""

import argparse
import os
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from action_recognition import CAUTION_ACTIONS, DANGER_ACTIONS, ActionRecognizer
from ultralytics import YOLO

# ==================== 骨骼连接关系 ====================
SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

FONT_PATH = None
for candidate in [
    "simhei.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
]:
    if os.path.exists(candidate):
        FONT_PATH = candidate
        break


def cv2_draw_chinese(img, text, pos, font_size=20, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_skeleton(img, keypoints, conf_thresh=0.5):
    pts = []
    for x, y, conf in keypoints:
        pts.append((int(x), int(y)) if conf > conf_thresh else None)
    for a, b in SKELETON:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        if p:
            cv2.circle(img, p, 3, (255, 0, 0), -1)


class VideoProcessor:
    """单路视频处理器：检测 + 跟踪 + 动作识别."""

    def __init__(self, source, pose_model, action_recognizer, seq_length=16):
        self.source = source
        self.pose_model = pose_model
        self.action_recognizer = action_recognizer
        self.seq_length = seq_length
        self.cap = cv2.VideoCapture(source)
        self.ok = self.cap.isOpened()
        self.track_history = defaultdict(lambda: deque(maxlen=seq_length * 2))
        self.track_last_action = {}
        self.frame_count = 0
        self.current_actions = []

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame_count += 1
        return frame

    def process(self, frame):
        h, w = frame.shape[:2]
        results = self.pose_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, verbose=False)
        annotated = frame.copy()
        current_track_ids = set()
        self.current_actions = []

        all_persons_info = []
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                keypoints = result.keypoints.data.cpu().numpy()
                for i, tid in enumerate(track_ids):
                    b = boxes[i]
                    all_persons_info.append(
                        {
                            "track_id": tid,
                            "bbox": b,
                            "kpts": keypoints[i],
                            "center": np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]),
                        }
                    )

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            keypoints = result.keypoints.data.cpu().numpy()

            for i, tid in enumerate(track_ids):
                current_track_ids.add(tid)
                kpts = keypoints[i]
                bbox = boxes[i]
                others = [p for p in all_persons_info if p["track_id"] != tid]
                self.track_history[tid].append(kpts.copy())

                if tid not in self.track_last_action:
                    self.track_last_action[tid] = "normal"

                action_en, action_cn, conf = self.action_recognizer.predict(
                    tid, list(self.track_history[tid]), bbox, (h, w), all_persons=others
                )
                self.track_last_action[tid] = action_en
                self.current_actions.append((action_en, action_cn, conf, tid, bbox))

                draw_skeleton(annotated, kpts)

                if action_en in DANGER_ACTIONS:
                    color = (0, 0, 255)
                elif action_en in CAUTION_ACTIONS:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {action_cn}"
                annotated = cv2_draw_chinese(annotated, label, (x1, max(y1 - 28, 0)), font_size=18, color=color)

        gone = set(self.track_history.keys()) - current_track_ids
        for tid in gone:
            del self.track_history[tid]
            self.track_last_action.pop(tid, None)
            self.action_recognizer.remove_track(tid)

        return annotated

    def close(self):
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="扶梯行人动作识别核心管线")
    parser.add_argument("--video", type=str, default="pose_test1.mp4", help="视频路径")
    parser.add_argument("--camera", action="store_true", help="使用摄像头")
    parser.add_argument("--direction", type=str, default="up", choices=["up", "down"], help="扶梯运行方向")
    parser.add_argument("--conf", type=float, default=0.5, help="检测置信度阈值")
    args = parser.parse_args()

    print("=" * 50)
    print("扶梯行人动作识别系统")
    print("YOLOv8-Pose + ByteTrack + ST-GCN（规则+深度学习混合）")
    print("=" * 50)

    print("\n[1/3] 加载 YOLOv8 姿态估计模型...")
    pose_model = YOLO("yolov8n-pose.pt")

    print("[2/3] 初始化动作识别器（规则引擎 + ST-GCN）...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_recognizer = ActionRecognizer(
        weight_path="st_gcn.kinetics.pt", escalator_direction=args.direction, device=device
    )
    print(f"    设备: {device}, 扶梯方向: {'上行' if args.direction == 'up' else '下行'}")

    print("[3/3] 模型加载完成")

    if args.camera:
        src = 0
        print("\n打开摄像头...")
    else:
        src = args.video
        print(f"\n打开视频: {src}")

    processor = VideoProcessor(src, pose_model, action_recognizer)
    if not processor.ok:
        print("错误: 无法打开视频源")
        return

    print("\n开始处理... 按 'q' 退出, 按 's' 截图")
    print("-" * 50)

    while True:
        frame = processor.read()
        if frame is None:
            break

        annotated = processor.process(frame)

        # 顶部统计
        actions = processor.current_actions
        if actions:
            counter = {}
            for _, act_cn, _, _, _ in actions:
                counter[act_cn] = counter.get(act_cn, 0) + 1
            sorted_acts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            info = " | ".join([f"{act}:{cnt}" for act, cnt in sorted_acts[:3]])
        else:
            info = "无行人"
        annotated = cv2_draw_chinese(annotated, f"行为统计: {info}", (10, 8), font_size=20, color=(255, 255, 255))

        cv2.imshow("扶梯行人动作识别", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"screenshot_{processor.frame_count:04d}.png", annotated)
            print(f"  截图已保存: screenshot_{processor.frame_count:04d}.png")

    processor.close()
    cv2.destroyAllWindows()
    print("-" * 50)
    print(f"处理完成，共 {processor.frame_count} 帧")


if __name__ == "__main__":
    main()
