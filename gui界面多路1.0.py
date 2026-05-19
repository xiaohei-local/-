"""
自动扶梯行人安全检测系统 — PyQt5 可视化界面
功能: 多路视频网格监控 + ROI区域检测 + 声音报警 + 干预按钮
"""
import sys
import os
import cv2
import numpy as np
import torch
import time
import threading
from collections import deque, defaultdict
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QSlider, QComboBox, QFileDialog,
    QMessageBox, QWidget, QListWidget, QStatusBar, QToolBar, QAction,
    QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

from database import init_db, verify_login, register_user, log_alert
from action_recognition import (
    ActionRecognizer, ACTION_CN, DANGER_ACTIONS, CAUTION_ACTIONS
)


def play_alarm_sound(stop_event):
    try:
        import winsound
        for _ in range(6):
            if stop_event.is_set():
                return
            winsound.Beep(2500, 200)
            if stop_event.is_set():
                return
            winsound.Beep(2000, 200)
    except ImportError:
        pass


SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4)
]

FONT_PATH = None
for c in ['simhei.ttf', 'C:/Windows/Fonts/simhei.ttf',
          'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc']:
    if os.path.exists(c):
        FONT_PATH = c
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


def draw_skeleton(img, kpts, conf_thresh=0.5):
    pts = []
    for x, y, c in kpts:
        pts.append((int(x), int(y)) if c > conf_thresh else None)
    for a, b in SKELETON:
        if pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        if p:
            cv2.circle(img, p, 3, (255, 0, 0), -1)


def draw_roi(img, roi):
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2)
        cv2.putText(img, 'ROI', (rx1, ry1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    return img


def is_in_roi(bbox_center, roi):
    if roi is None:
        return True
    rx1, ry1, rx2, ry2 = roi
    cx, cy = bbox_center
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


# ==================== 登录窗口 ====================
class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动扶梯安全检测系统 - 登录")
        self.setFixedSize(420, 320)
        self.setStyleSheet(self._style())
        self._init_ui()

    def _style(self):
        return """
        QDialog { background: #f0f2f5; }
        QLabel#title { font-size: 20px; font-weight: bold; color: #1a1a2e; }
        QLineEdit { padding: 10px 14px; border: 2px solid #d1d5db; border-radius: 8px;
                    font-size: 14px; background: white; }
        QLineEdit:focus { border-color: #4f46e5; }
        QPushButton { padding: 10px; border-radius: 8px; font-size: 14px; font-weight: bold;
                      color: white; border: none; }
        QPushButton#login_btn { background: #4f46e5; }
        QPushButton#login_btn:hover { background: #4338ca; }
        QPushButton#register_btn { background: #10b981; }
        QPushButton#register_btn:hover { background: #059669; }
        """

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(40, 30, 40, 30)
        title = QLabel("自动扶梯行人安全检测系统")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(10)
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("用户名")
        layout.addWidget(self.user_edit)
        self.pass_edit = QLineEdit()
        self.pass_edit.setPlaceholderText("密码")
        self.pass_edit.setEchoMode(QLineEdit.Password)
        self.pass_edit.returnPressed.connect(self._login)
        layout.addWidget(self.pass_edit)
        btn_layout = QHBoxLayout()
        self.login_btn = QPushButton("登 录")
        self.login_btn.setObjectName("login_btn")
        self.login_btn.clicked.connect(self._login)
        btn_layout.addWidget(self.login_btn)
        self.register_btn = QPushButton("注 册")
        self.register_btn.setObjectName("register_btn")
        self.register_btn.clicked.connect(self._register)
        btn_layout.addWidget(self.register_btn)
        layout.addLayout(btn_layout)
        self.msg_label = QLabel("")
        self.msg_label.setAlignment(Qt.AlignCenter)
        self.msg_label.setStyleSheet("color: #ef4444; font-size: 13px;")
        layout.addWidget(self.msg_label)
        self.setLayout(layout)

    def _login(self):
        username = self.user_edit.text().strip()
        password = self.pass_edit.text().strip()
        if not username or not password:
            self.msg_label.setText("请输入用户名和密码")
            return
        ok, role = verify_login(username, password)
        if ok:
            self.accept()
        else:
            self.msg_label.setText("用户名或密码错误")

    def _register(self):
        username = self.user_edit.text().strip()
        password = self.pass_edit.text().strip()
        if not username or not password:
            self.msg_label.setText("请输入用户名和密码")
            return
        if len(password) < 4:
            self.msg_label.setText("密码长度至少4位")
            return
        ok, msg = register_user(username, password)
        if ok:
            self.msg_label.setStyleSheet("color: #10b981; font-size: 13px;")
            self.msg_label.setText(msg + "，请登录")
        else:
            self.msg_label.setStyleSheet("color: #ef4444; font-size: 13px;")
            self.msg_label.setText(msg)


# ==================== 可拖拽ROI的视频标签 ====================
class VideoLabel(QLabel):
    roi_changed = pyqtSignal(int, object)

    def __init__(self, video_idx, parent=None):
        super().__init__(parent)
        self.video_idx = video_idx
        self.roi = None
        self.drawing_roi = False
        self.roi_start = None
        self.roi_end = None
        self.roi_mode = False
        self._orig_size = (640, 480)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 200)
        self.setStyleSheet("""
            QLabel { background: #1e1e2e; color: #a0a0b0;
                     border: 2px solid #3b3b52; border-radius: 4px; font-size: 14px; }
        """)

    def set_roi_mode(self, enabled):
        self.roi_mode = enabled
        if not enabled:
            self.drawing_roi = False

    def clear_roi(self):
        self.roi = None
        self.roi_changed.emit(self.video_idx, None)

    def mousePressEvent(self, event):
        if self.roi_mode and event.button() == Qt.LeftButton:
            self.drawing_roi = True
            self.roi_start = (event.x(), event.y())
            self.roi_end = (event.x(), event.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.roi_mode and self.drawing_roi:
            self.roi_end = (event.x(), event.y())
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.roi_mode and self.drawing_roi and event.button() == Qt.LeftButton:
            self.drawing_roi = False
            self.roi_end = (event.x(), event.y())
            if self.pixmap():
                pw = self.pixmap().width()
                ph = self.pixmap().height()
                lw = self.width()
                lh = self.height()
                if lw > 0 and lh > 0:
                    scale_w = pw / lw
                    scale_h = ph / lh
                    sx, sy = self.roi_start
                    ex, ey = self.roi_end
                    x1 = int(min(sx, ex) * scale_w)
                    y1 = int(min(sy, ey) * scale_h)
                    x2 = int(max(sx, ex) * scale_w)
                    y2 = int(max(sy, ey) * scale_h)
                    if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                        self.roi = (x1, y1, x2, y2)
                    else:
                        self.roi = None
                    self.roi_changed.emit(self.video_idx, self.roi)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.roi_mode and self.drawing_roi and self.roi_start and self.roi_end:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 200, 0), 2, Qt.DashLine))
            sx, sy = self.roi_start
            ex, ey = self.roi_end
            painter.drawRect(QRect(min(sx, ex), min(sy, ey), abs(ex - sx), abs(ey - sy)))
            painter.end()


# ==================== 主窗口（QTimer 主线程处理，与命令行版一致） ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动扶梯行人安全检测系统")
        self.setMinimumSize(1280, 760)

        self.timer = QTimer()
        self.timer.timeout.connect(self._process_tick)

        self.running = False
        self.pose_model = None
        self.action_recognizer = None
        self.caps = []
        self.names = []
        self.rois = {}
        self.track_hists = []
        self.track_last = []
        self.alert_logs = []
        self.counters = []
        self.fps_deques = []
        self.prev_ts = []
        self.rr = 0
        self.conf_threshold = 0.5
        self.escalator_direction = 'up'
        self.seq_length = 16

        self.video_labels = []
        self.video_sources = []
        self.roi_mode = False
        self.emergency_stop_active = False

        self._init_ui()
        self._init_toolbar()
        self._init_statusbar()

    # ========== UI 初始化 ==========
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        left_panel = QVBoxLayout()
        self.video_grid = QGridLayout()
        self.video_grid.setSpacing(4)
        self._setup_video_grid(1)
        grid_widget = QWidget()
        grid_widget.setLayout(self.video_grid)
        left_panel.addWidget(grid_widget, 1)

        bottom_bar = QHBoxLayout()
        self.btn_add_video = QPushButton("添加视频源")
        self.btn_add_video.clicked.connect(self._add_video_source)
        bottom_bar.addWidget(self.btn_add_video)
        self.btn_remove_video = QPushButton("移除最后一路")
        self.btn_remove_video.clicked.connect(self._remove_video_source)
        bottom_bar.addWidget(self.btn_remove_video)
        self.btn_roi_mode = QPushButton("绘制ROI区域")
        self.btn_roi_mode.setCheckable(True)
        self.btn_roi_mode.clicked.connect(self._toggle_roi_mode)
        bottom_bar.addWidget(self.btn_roi_mode)
        self.btn_clear_roi = QPushButton("清除ROI")
        self.btn_clear_roi.clicked.connect(self._clear_all_roi)
        bottom_bar.addWidget(self.btn_clear_roi)
        bottom_bar.addStretch()
        left_panel.addLayout(bottom_bar)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        right_widget = QWidget()
        right_widget.setFixedWidth(420)
        right_widget.setLayout(right_panel)

        ctrl_group = QGroupBox("控制面板")
        ctrl_group.setStyleSheet(self._group_style())
        ctrl_layout = QGridLayout(ctrl_group)
        ctrl_layout.setSpacing(8)

        self.source_list = QListWidget()
        self.source_list.setMaximumHeight(80)
        self.source_list.setStyleSheet("font-size: 11px;")
        ctrl_layout.addWidget(QLabel("视频源列表:"), 0, 0, 1, 2)
        ctrl_layout.addWidget(self.source_list, 1, 0, 1, 2)

        self.btn_start = QPushButton("开始检测")
        self.btn_start.setStyleSheet(self._btn_style('#10b981', '#059669'))
        self.btn_start.clicked.connect(self._start)
        ctrl_layout.addWidget(self.btn_start, 2, 0)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self._btn_style('#ef4444', '#dc2626'))
        self.btn_stop.clicked.connect(self._stop)
        ctrl_layout.addWidget(self.btn_stop, 2, 1)

        ctrl_layout.addWidget(QLabel("置信度阈值:"), 3, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(50)
        self.conf_slider.setTickInterval(10)
        self.conf_label = QLabel("0.50")
        ctrl_layout.addWidget(self.conf_slider, 3, 1)
        ctrl_layout.addWidget(self.conf_label, 4, 1)
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v / 100:.2f}"))

        ctrl_layout.addWidget(QLabel("扶梯方向:"), 4, 0)
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(["上行", "下行"])
        ctrl_layout.addWidget(self.dir_combo, 4, 1)

        self.btn_call_staff = QPushButton("呼叫工作人员")
        self.btn_call_staff.setStyleSheet(self._btn_style('#f59e0b', '#d97706'))
        self.btn_call_staff.clicked.connect(self._call_staff)
        ctrl_layout.addWidget(self.btn_call_staff, 5, 0)

        self.btn_emergency = QPushButton("电梯急停")
        self.btn_emergency.setStyleSheet(self._btn_style('#ef4444', '#dc2626'))
        self.btn_emergency.clicked.connect(self._emergency_stop)
        ctrl_layout.addWidget(self.btn_emergency, 5, 1)

        self.btn_dismiss = QPushButton("解除警报")
        self.btn_dismiss.setStyleSheet(self._btn_style('#6b7280', '#4b5563'))
        self.btn_dismiss.clicked.connect(self._dismiss_alert)
        ctrl_layout.addWidget(self.btn_dismiss, 6, 0, 1, 2)

        right_panel.addWidget(ctrl_group)

        alert_group = QGroupBox("实时告警")
        alert_group.setStyleSheet(self._group_style())
        alert_layout = QVBoxLayout(alert_group)
        self.alert_table = QTableWidget(0, 4)
        self.alert_table.setHorizontalHeaderLabels(["时间", "行人ID", "行为", "置信度"])
        self.alert_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.alert_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.alert_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.alert_table.verticalHeader().setVisible(False)
        self.alert_table.setMaximumHeight(180)
        self.alert_table.setStyleSheet("""
            QTableWidget { background: white; gridline-color: #e5e7eb; font-size: 13px; }
            QHeaderView::section { background: #f3f4f6; padding: 6px; font-weight: bold; border: none; }
        """)
        alert_layout.addWidget(self.alert_table)
        right_panel.addWidget(alert_group)

        stats_group = QGroupBox("行为统计")
        stats_group.setStyleSheet(self._group_style())
        self.stats_layout = QGridLayout(stats_group)
        self.stats_layout.setSpacing(4)
        self.stats_labels = {}
        for idx, (key, cn, color) in enumerate([
            ('normal', '正常', '#10b981'), ('running', '奔跑', '#ef4444'),
            ('reverse', '逆行', '#ef4444'), ('falling', '摔倒', '#ef4444'),
            ('loitering', '滞留', '#ef4444'), ('sitting', '坐下', '#f59e0b'),
        ]):
            row, col = divmod(idx, 2)
            lbl = QLabel(f"{cn}: 0")
            lbl.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")
            self.stats_labels[key] = lbl
            self.stats_layout.addWidget(lbl, row, col)
        right_panel.addWidget(stats_group)

        right_panel.addStretch()
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(right_widget)

    def _setup_video_grid(self, count):
        for lbl in self.video_labels:
            self.video_grid.removeWidget(lbl)
            lbl.deleteLater()
        self.video_labels.clear()
        if count <= 2:
            cols, rows = count, 1
        else:
            cols, rows = 2, (count + 1) // 2
        for i in range(count):
            vl = VideoLabel(i)
            vl.setText(f"视频源 {i + 1}\n请添加视频文件")
            self.video_labels.append(vl)
            row, col = divmod(i, cols)
            self.video_grid.addWidget(vl, row, col)
        for i, (name, path, is_cam) in enumerate(self.video_sources):
            if i < len(self.video_labels):
                self.video_labels[i].setText(f"{name}\n待开始检测")

    def _add_video_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)")
        if not path:
            return
        self.video_sources.append((os.path.basename(path), path, False))
        self._setup_video_grid(min(max(len(self.video_sources), 1), 4))
        self._update_source_list()

    def _remove_video_source(self):
        if self.video_sources:
            self.video_sources.pop()
            self._setup_video_grid(max(len(self.video_sources), 1))
            self._update_source_list()

    def _update_source_list(self):
        self.source_list.clear()
        for name, path, is_cam in self.video_sources:
            self.source_list.addItem(f"{'[CAM]' if is_cam else '[VID]'} {name}")

    def _toggle_roi_mode(self, checked):
        self.roi_mode = checked
        for vl in self.video_labels:
            vl.set_roi_mode(checked)
        self.status_label.setText("ROI模式: 拖拽绘制检测区域" if checked else "ROI模式已关闭")

    def _clear_all_roi(self):
        for vl in self.video_labels:
            vl.clear_roi()

    def _init_toolbar(self):
        toolbar = QToolBar("工具栏")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        act_screenshot = QAction("截图", self)
        act_screenshot.triggered.connect(self._screenshot)
        toolbar.addAction(act_screenshot)
        act_export = QAction("导出报告", self)
        act_export.triggered.connect(self._export_report)
        toolbar.addAction(act_export)
        toolbar.addSeparator()
        act_about = QAction("关于", self)
        act_about.triggered.connect(self._about)
        toolbar.addAction(act_about)

    def _init_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.fps_label = QLabel("FPS: --")
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.fps_label)

    def _group_style(self):
        return """
        QGroupBox { font-weight: bold; font-size: 14px; border: 1px solid #d1d5db;
                    border-radius: 6px; margin-top: 10px; padding-top: 16px; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        """

    def _btn_style(self, bg, hover):
        return f"""
        QPushButton {{ background: {bg}; color: white; border: none;
                      padding: 10px; border-radius: 6px; font-size: 13px; font-weight: bold; }}
        QPushButton:hover {{ background: {hover}; }}
        QPushButton:disabled {{ background: #9ca3af; }}
        """

    # ========== 控制逻辑 ==========
    def _start(self):
        if not self.video_sources:
            QMessageBox.warning(self, "提示", "请先添加至少一个视频源")
            return

        self.status_label.setText("加载模型中...")
        self.status_label.setStyleSheet("color: #f59e0b; font-weight: bold;")
        QApplication.processEvents()

        # 主线程加载模型（与命令行版 YOLO + ByteTrack + ST-GCN.py 一致）
        print("[GUI] 加载 YOLO...")
        self.pose_model = YOLO('yolov8n-pose.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.escalator_direction = 'up' if self.dir_combo.currentIndex() == 0 else 'down'
        print(f"[GUI] 加载 ST-GCN (device={device})...")
        self.action_recognizer = ActionRecognizer(
            weight_path='st_gcn.kinetics.pt',
            escalator_direction=self.escalator_direction,
            device=device
        )
        self.conf_threshold = self.conf_slider.value() / 100.0

        # 主线程打开视频（与命令行版一致）
        num = len(self.video_sources)
        self.caps = []
        self.names = []
        self.rois = {}
        self.track_hists = []
        self.track_last = []
        self.alert_logs = []
        self.counters = []
        self.fps_deques = []
        self.prev_ts = []

        for i, (name, src, is_cam) in enumerate(self.video_sources):
            cap_src = 0 if is_cam else src
            print(f"[GUI] 打开视频: {name}")
            cap = cv2.VideoCapture(cap_src)
            self.caps.append(cap if cap.isOpened() else None)
            self.names.append(name)
            self.track_hists.append(defaultdict(lambda: deque(maxlen=self.seq_length * 2)))
            self.track_last.append({})
            self.alert_logs.append(set())
            self.counters.append(defaultdict(int))
            self.fps_deques.append(deque(maxlen=30))
            self.prev_ts.append(time.time())
            if self.video_labels[i].roi:
                self.rois[i] = self.video_labels[i].roi

        self.rr = 0
        self.running = True
        self.emergency_stop_active = False
        self.timer.start(30)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_add_video.setEnabled(False)
        self.btn_remove_video.setEnabled(False)
        self.btn_roi_mode.setEnabled(False)
        self.btn_clear_roi.setEnabled(False)
        self.status_label.setText("运行中...")
        self.status_label.setStyleSheet("")
        self.alert_table.setRowCount(0)

    def _stop(self):
        self.running = False
        self.timer.stop()
        for c in self.caps:
            if c:
                try:
                    c.release()
                except Exception:
                    pass
        self.caps.clear()
        self._reset_ui_state()

    def _process_tick(self):
        # 防重入：上一帧处理未完成时跳过本次 tick
        if not self.running or not self.caps:
            return
        if getattr(self, '_busy', False):
            return
        self._busy = True

        num = len(self.caps)
        if not any(c and c.isOpened() for c in self.caps):
            self._stop()
            self.status_label.setText("处理完成")
            self._busy = False
            return

        idx = self.rr % num
        self.rr += 1
        cap = self.caps[idx]
        if cap is None or not cap.isOpened():
            self._busy = False
            return

        ret, frame = cap.read()
        if not ret:
            self.caps[idx].release()
            self.caps[idx] = None
            self._busy = False
            return

        h, w = frame.shape[:2]
        roi = self.rois.get(idx, None)

        results = self.pose_model.track(frame, persist=True, tracker='bytetrack.yaml',
                                        conf=self.conf_threshold, verbose=False)
        annotated = frame.copy()
        cur_ids = set()

        all_persons = []
        for r in results:
            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                tids = r.boxes.id.int().cpu().tolist()
                kpts_arr = r.keypoints.data.cpu().numpy()
                for j, tid in enumerate(tids):
                    b = boxes[j]
                    ctr = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                    if not is_in_roi(ctr, roi):
                        continue
                    all_persons.append({
                        'track_id': tid, 'bbox': b, 'kpts': kpts_arr[j], 'center': ctr
                    })

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            tids = r.boxes.id.int().cpu().tolist()
            kpts_arr = r.keypoints.data.cpu().numpy()

            for j, tid in enumerate(tids):
                b = boxes[j]
                ctr = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                if not is_in_roi(ctr, roi):
                    continue

                cur_ids.add(tid)
                kpts = kpts_arr[j]
                bbox = boxes[j]
                others = [p for p in all_persons if p['track_id'] != tid]
                self.track_hists[idx][tid].append(kpts.copy())

                if tid not in self.track_last[idx]:
                    self.track_last[idx][tid] = 'normal'

                action_en, action_cn, conf = self.action_recognizer.predict(
                    tid, list(self.track_hists[idx][tid]), bbox, (h, w),
                    all_persons=others
                )
                self.track_last[idx][tid] = action_en
                self.counters[idx][action_cn] += 1

                if action_en in DANGER_ACTIONS or action_en in CAUTION_ACTIONS:
                    ak = (tid, action_en)
                    if ak not in self.alert_logs[idx]:
                        self.alert_logs[idx].add(ak)
                        if len(self.alert_logs[idx]) > 500:
                            self.alert_logs[idx].clear()
                        ts = datetime.now().strftime("%H:%M:%S")
                        self._on_alert(idx, ts, tid, action_cn, conf)
                        log_alert(self.names[idx], tid, action_en, action_cn, conf, 0)

                draw_skeleton(annotated, kpts)
                if action_en in DANGER_ACTIONS:
                    color = (0, 0, 255)
                elif action_en in CAUTION_ACTIONS:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f'ID:{tid} {action_cn}'
                annotated = cv2_draw_chinese(annotated, label, (x1, max(y1 - 28, 0)),
                                             font_size=18, color=color)

        # 清理
        gone = set(self.track_hists[idx].keys()) - cur_ids
        for tid in gone:
            del self.track_hists[idx][tid]
            self.track_last[idx].pop(tid, None)
            self.action_recognizer.remove_track(tid)

        annotated = draw_roi(annotated, roi)
        total = len(cur_ids)
        annotated = cv2_draw_chinese(
            annotated,
            f"[{self.names[idx]}] 行人:{total} | {'上行' if self.escalator_direction == 'up' else '下行'}",
            (10, 8), font_size=18, color=(255, 255, 255))

        # FPS
        now = time.time()
        dt = max(now - self.prev_ts[idx], 0.001)
        self.fps_deques[idx].append(1.0 / dt)
        self.prev_ts[idx] = now
        afps = sum(self.fps_deques[idx]) / len(self.fps_deques[idx])
        self.fps_label.setText(f"FPS: {afps:.1f}")

        # 显示
        if idx < len(self.video_labels):
            vl = self.video_labels[idx]
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            hh, ww, ch = rgb.shape
            # copy() 断开与 numpy 共享内存，避免 Qt 在渲染后被 numpy 释放导致崩溃
            qimg = QImage(rgb.copy().data, ww, hh, ch * ww, QImage.Format_RGB888)
            vl._orig_size = (ww, hh)
            scaled = qimg.scaled(vl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            vl.setPixmap(QPixmap.fromImage(scaled))

        # 合并统计
        merged = defaultdict(int)
        for ctr in self.counters:
            for k, v in ctr.items():
                merged[k] += v
        for key, lbl in self.stats_labels.items():
            cn = ACTION_CN.get(key, key)
            lbl.setText(f"{cn}: {merged.get(cn, 0)}")

        self._busy = False

    # ========== 告警与干预 ==========
    def _on_alert(self, video_idx, timestamp, track_id, action_cn, confidence):
        if self.emergency_stop_active:
            return
        row = self.alert_table.rowCount()
        self.alert_table.insertRow(0)
        self.alert_table.setItem(0, 0, QTableWidgetItem(timestamp))
        self.alert_table.setItem(0, 1, QTableWidgetItem(str(track_id)))
        self.alert_table.setItem(0, 2, QTableWidgetItem(action_cn))
        self.alert_table.setItem(0, 3, QTableWidgetItem(f"{confidence:.2f}"))
        if action_cn in ['奔跑', '逆行', '摔倒', '滞留']:
            for c in range(4):
                self.alert_table.item(0, c).setForeground(QColor('#ef4444'))
            self._start_alarm()
        elif action_cn == '坐下':
            for c in range(4):
                self.alert_table.item(0, c).setForeground(QColor('#f59e0b'))
        while self.alert_table.rowCount() > 100:
            self.alert_table.removeRow(self.alert_table.rowCount() - 1)

    def _start_alarm(self):
        if self.emergency_stop_active:
            return
        if hasattr(self, '_alarm_thread') and self._alarm_thread and self._alarm_thread.is_alive():
            return
        self._alarm_stop = threading.Event()
        self._alarm_thread = threading.Thread(target=play_alarm_sound, args=(self._alarm_stop,), daemon=True)
        self._alarm_thread.start()

    def _stop_alarm(self):
        if hasattr(self, '_alarm_stop'):
            self._alarm_stop.set()

    def _call_staff(self):
        self._stop_alarm()
        self.status_label.setText("[!] 已呼叫工作人员 - 请等待响应")
        self.status_label.setStyleSheet("color: #f59e0b; font-weight: bold;")

    def _emergency_stop(self):
        self._stop_alarm()
        self.emergency_stop_active = True
        self.status_label.setText("[!!!] 电梯已急停 - 检测继续运行中")
        self.status_label.setStyleSheet("color: #ef4444; font-weight: bold;")

    def _dismiss_alert(self):
        self._stop_alarm()
        self.emergency_stop_active = False
        self.status_label.setText("就绪 - 警报已解除")
        self.status_label.setStyleSheet("color: #10b981; font-weight: bold;")

    def _reset_ui_state(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_add_video.setEnabled(True)
        self.btn_remove_video.setEnabled(True)
        self.btn_roi_mode.setEnabled(True)
        self.btn_clear_roi.setEnabled(True)
        self.fps_label.setText("FPS: --")
        self.status_label.setStyleSheet("")
        self.emergency_stop_active = False

    def _screenshot(self):
        for i, vl in enumerate(self.video_labels):
            if vl.pixmap():
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                vl.pixmap().save(f"screenshot_ch{i}_{ts}.png")
        self.status_label.setText("截图已保存")

    def _export_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出报告", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "文本文件 (*.txt)")
        if not path:
            return
        try:
            from database import get_recent_alerts, get_alert_stats
            with open(path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("自动扶梯行人安全检测系统 - 检测报告\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"视频源数量: {len(self.video_sources)}\n")
                f.write("=" * 50 + "\n\n[行为统计]\n")
                for action, cn, count in get_alert_stats():
                    f.write(f"  {cn}: {count} 次\n")
                f.write("\n[告警记录]\n")
                for t, tid, act_cn, conf in get_recent_alerts(200):
                    f.write(f"  {t} | ID:{tid} | {act_cn} | 置信度:{conf:.2f}\n")
            self.status_label.setText(f"报告已导出: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")

    def _about(self):
        QMessageBox.about(self, "关于",
                          "自动扶梯行人安全检测系统 v2.0\n\n"
                          "技术栈: YOLOv8-Pose + ByteTrack + ST-GCN\n"
                          "使用规则判别 + 深度学习混合方案\n\n"
                          "功能: 多路视频监控 | ROI区域检测 | 声音报警 | 干预控制")

    def closeEvent(self, event):
        self._stop_alarm()
        self.running = False
        self.timer.stop()
        for c in self.caps:
            if c:
                try:
                    c.release()
                except Exception:
                    pass
        event.accept()


def main():
    init_db()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Microsoft YaHei", 10))
    login = LoginWindow()
    if login.exec_() != QDialog.Accepted:
        sys.exit(0)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
