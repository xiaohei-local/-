"""
自动扶梯行人安全检测系统 — PyQt5 可视化界面 (Dark Dashboard)
功能: 多路视频网格监控 + ROI区域检测 + 声音报警 + 干预按钮.
"""

import os
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from action_recognition import ACTION_CN, CAUTION_ACTIONS, DANGER_ACTIONS, ActionRecognizer
from database import init_db, log_alert, register_user, verify_login
from ultralytics import YOLO


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
for c in ["simhei.ttf", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"]:
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
        cv2.putText(img, "ROI", (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    return img


def is_in_roi(bbox_center, roi):
    if roi is None:
        return True
    rx1, ry1, rx2, ry2 = roi
    cx, cy = bbox_center
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


# ==================== Dark Theme ====================
DARK_QSS = """
QMainWindow { background: #0f0f1a; }
QWidget { color: #e0e0e0; font-family: "Microsoft YaHei"; }

/* Panel cards */
QWidget#leftPanel, QWidget#rightPanel, QWidget#bottomPanel {
    background: #1a1a2e;
}

/* Group boxes */
QGroupBox {
    font-weight: bold; font-size: 13px; color: #c0c0d0;
    border: 1px solid #2a2a4a; border-radius: 8px;
    margin-top: 14px; padding: 16px 10px 10px 10px;
    background: #1a1a2e;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 14px; padding: 0 8px;
    color: #8b8ba0;
}

/* Buttons */
QPushButton {
    background: #2a2a4a; color: #d0d0e0; border: 1px solid #3b3b52;
    padding: 8px 14px; border-radius: 6px; font-size: 13px;
}
QPushButton:hover { background: #35355a; border-color: #4f46e5; }
QPushButton:pressed { background: #1e1e38; }
QPushButton:disabled { background: #1a1a2e; color: #606070; border-color: #2a2a3a; }

/* Line edits */
QLineEdit {
    padding: 10px 14px; border: 2px solid #3b3b52; border-radius: 8px;
    font-size: 14px; background: #16213e; color: #e0e0e0;
}
QLineEdit:focus { border-color: #4f46e5; }

/* Table */
QTableWidget {
    background: #16213e; color: #d0d0e0;
    gridline-color: #2a2a4a; border: 1px solid #2a2a4a; border-radius: 6px;
    font-size: 13px;
}
QTableWidget::item { padding: 5px; }
QTableWidget::item:selected { background: #4f46e5; }
QHeaderView::section {
    background: #1a1a2e; color: #8b8ba0; padding: 8px;
    font-weight: bold; border: none; border-bottom: 2px solid #2a2a4a;
}

/* Slider */
QSlider::groove:horizontal { background: #2a2a4a; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #4f46e5; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px;
}
QSlider::handle:horizontal:hover { background: #6366f1; }
QSlider::sub-page:horizontal { background: #4f46e5; border-radius: 3px; }

/* List */
QListWidget {
    background: #16213e; color: #d0d0e0; border: 1px solid #2a2a4a;
    border-radius: 6px; font-size: 12px;
}
QListWidget::item { padding: 7px 10px; border-bottom: 1px solid #1e1e38; }
QListWidget::item:hover { background: #2a2a4a; }
QListWidget::item:selected { background: #4f46e5; color: #ffffff; }

/* Combo box */
QComboBox {
    background: #2a2a4a; color: #e0e0e0; border: 1px solid #3b3b52;
    border-radius: 6px; padding: 6px 12px; font-size: 13px;
}
QComboBox:hover { border-color: #4f46e5; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #1a1a2e; color: #e0e0e0; selection-background-color: #4f46e5;
    border: 1px solid #2a2a4a; outline: none;
}

/* Splitter */
QSplitter::handle { background: #2a2a4a; }
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }
QSplitter::handle:hover { background: #4f46e5; }

/* Scroll bars */
QScrollBar:vertical {
    background: #0f0f1a; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #3b3b52; border-radius: 4px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #4f46e5; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #0f0f1a; height: 8px; border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #3b3b52; border-radius: 4px; min-width: 24px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* Scroll area */
QScrollArea { border: none; background: transparent; }

/* Status bar */
QStatusBar { background: #1a1a2e; color: #8b8ba0; border-top: 1px solid #2a2a4a; font-size: 12px; }
QStatusBar QLabel { color: #8b8ba0; border: none; padding: 0 8px; }

/* ComboBox dropdown items */
QComboBox QAbstractItemView::item {
    padding: 7px 14px; color: #e0e0e0;
}
QComboBox QAbstractItemView::item:hover {
    background: #2a2a4a;
}
QComboBox QAbstractItemView::item:selected {
    background: #4f46e5; color: #ffffff;
}

/* Message box */
QMessageBox {
    background: #1a1a2e; color: #e0e0e0;
}
QMessageBox QLabel {
    color: #e0e0e0; font-size: 13px; min-width: 240px;
}
QMessageBox QPushButton {
    background: #2a2a4a; color: #d0d0e0; border: 1px solid #3b3b52;
    padding: 7px 22px; border-radius: 6px; font-size: 12px; min-width: 64px;
}
QMessageBox QPushButton:hover {
    background: #35355a; border-color: #4f46e5;
}

/* Tool tips */
QToolTip {
    background: #2a2a4a; color: #e0e0e0; border: 1px solid #3b3b52;
    border-radius: 4px; padding: 4px 8px; font-size: 12px;
}
"""


# ==================== 登录窗口 ====================
class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动扶梯安全检测系统 - 登录")
        self.setFixedSize(420, 360)
        self.setStyleSheet(self._style())
        self._init_ui()

    def _style(self):
        return (
            DARK_QSS
            + """
        QDialog { background: #1a1a2e; }
        QLabel#title { font-size: 22px; font-weight: bold; color: #e0e0e0; }
        QLabel#subtitle { font-size: 12px; color: #8b8ba0; }
        """
        )

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(14)
        layout.setContentsMargins(44, 36, 44, 36)

        title = QLabel("自动扶梯行人安全检测系统")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Escalator Pedestrian Safety Detection System")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        layout.addSpacing(8)

        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("用户名")
        layout.addWidget(self.user_edit)

        self.pass_edit = QLineEdit()
        self.pass_edit.setPlaceholderText("密码")
        self.pass_edit.setEchoMode(QLineEdit.Password)
        self.pass_edit.returnPressed.connect(self._login)
        layout.addWidget(self.pass_edit)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.login_btn = QPushButton("登 录")
        self.login_btn.setStyleSheet(
            "QPushButton { background: #4f46e5; color: white; border: none; "
            "padding: 12px; border-radius: 8px; font-size: 15px; font-weight: bold; }"
            "QPushButton:hover { background: #4338ca; }"
        )
        self.login_btn.clicked.connect(self._login)
        btn_layout.addWidget(self.login_btn)

        self.register_btn = QPushButton("注 册")
        self.register_btn.setStyleSheet(
            "QPushButton { background: #10b981; color: white; border: none; "
            "padding: 12px; border-radius: 8px; font-size: 15px; font-weight: bold; }"
            "QPushButton:hover { background: #059669; }"
        )
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
        ok, _role = verify_login(username, password)
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
        self.setMinimumSize(280, 180)
        self.setStyleSheet(
            "QLabel { background: #0a0a14; color: #6b6b80; "
            "border: 2px solid #252540; border-radius: 6px; font-size: 14px; }"
        )

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


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自动扶梯行人安全检测系统")
        self.setMinimumSize(1280, 760)
        self.resize(1440, 860)

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
        self.counters = []
        self.fps_deques = []
        self.prev_ts = []
        self.rr = 0
        self.conf_threshold = 0.5
        self.skeleton_conf = 0.3
        self.frame_skip = 1
        self.alert_cooldown = 3
        self.escalator_direction = "up"
        self.seq_length = 16

        self._alert_cooldown_times = {}
        self._frame_counter = 0

        self.video_labels = []
        self.video_sources = []
        self.roi_mode = False
        self.emergency_stop_active = False

        self.left_panel_visible = True
        self.layout_mode = "auto"
        self.active_video_idx = 0

        self._init_ui()
        self._init_statusbar()

    # ========== UI 初始化 ==========
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.setSpacing(0)

        # ---- header bar ----
        main_vbox.addWidget(self._create_header())

        # ---- main splitter (content | bottom alert) ----
        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setHandleWidth(3)

        # ---- content splitter (left | center | right) ----
        self.content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter.setHandleWidth(3)

        self.left_panel_widget = self._create_left_panel()
        self.content_splitter.addWidget(self.left_panel_widget)

        self.video_grid_widget = QWidget()
        self.video_grid_widget.setStyleSheet("background: #0a0a14; border-radius: 6px;")
        self.video_grid = QGridLayout()
        self.video_grid.setSpacing(3)
        self.video_grid.setContentsMargins(3, 3, 3, 3)
        self.video_grid_widget.setLayout(self.video_grid)
        self._setup_video_grid(1, "auto")
        self.content_splitter.addWidget(self.video_grid_widget)

        self.right_panel_scroll = self._create_right_panel()
        self.content_splitter.addWidget(self.right_panel_scroll)

        self.content_splitter.setSizes([200, 620, 320])
        self.content_splitter.setStretchFactor(0, 0)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setStretchFactor(2, 0)

        self.main_splitter.addWidget(self.content_splitter)

        # ---- bottom alert panel ----
        self.bottom_panel_widget = self._create_bottom_panel()
        self.main_splitter.addWidget(self.bottom_panel_widget)

        self.main_splitter.setSizes([600, 180])
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)

        main_vbox.addWidget(self.main_splitter)

    def _create_header(self):
        """Top toolbar bar."""
        header = QWidget()
        header.setFixedHeight(44)
        header.setStyleSheet("QWidget { background: #1a1a2e; border-bottom: 1px solid #2a2a4a; }")

        h = QHBoxLayout(header)
        h.setContentsMargins(14, 6, 14, 6)
        h.setSpacing(6)

        # left-panel toggle (always visible)
        self.btn_toggle_left = QPushButton("◀")
        self.btn_toggle_left.setFixedSize(28, 28)
        self.btn_toggle_left.setToolTip("折叠/展开视频列表面板")
        self.btn_toggle_left.clicked.connect(self._toggle_left_panel)
        self.btn_toggle_left.setStyleSheet(
            "QPushButton { background: #252540; color: #8b8ba0; border: 1px solid #333350; "
            "border-radius: 5px; font-size: 12px; }"
            "QPushButton:hover { background: #333358; color: #fff; border-color: #4f46e5; }"
        )
        h.addWidget(self.btn_toggle_left)

        h.addSpacing(8)

        title = QLabel("自动扶梯安全检测系统")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #e0e0e0; border: none; padding-right: 20px;")
        h.addWidget(title)

        h.addSpacing(12)

        btn_css = (
            "QPushButton { background: #252540; color: #c0c0d0; border: 1px solid #333350; "
            "padding: 4px 14px; border-radius: 5px; font-size: 12px; }"
            "QPushButton:hover { background: #333358; border-color: #4f46e5; color: #fff; }"
            "QPushButton:checked { background: #4f46e5; border-color: #4f46e5; color: #fff; }"
            "QPushButton:disabled { background: #1a1a2e; color: #505060; border-color: #2a2a38; }"
        )

        self.btn_add_video = QPushButton("+ 添加视频")
        self.btn_add_video.setStyleSheet(btn_css)
        self.btn_add_video.clicked.connect(self._add_video_source)
        h.addWidget(self.btn_add_video)

        self.btn_remove_video = QPushButton("- 移除视频")
        self.btn_remove_video.setStyleSheet(btn_css)
        self.btn_remove_video.clicked.connect(self._remove_video_source)
        h.addWidget(self.btn_remove_video)

        h.addSpacing(6)

        self.btn_roi_mode = QPushButton("ROI绘制")
        self.btn_roi_mode.setCheckable(True)
        self.btn_roi_mode.setStyleSheet(btn_css)
        self.btn_roi_mode.clicked.connect(self._toggle_roi_mode)
        h.addWidget(self.btn_roi_mode)

        self.btn_clear_roi = QPushButton("清除ROI")
        self.btn_clear_roi.setStyleSheet(btn_css)
        self.btn_clear_roi.clicked.connect(self._clear_all_roi)
        h.addWidget(self.btn_clear_roi)

        h.addSpacing(6)

        self.btn_screenshot = QPushButton("截图")
        self.btn_screenshot.setStyleSheet(btn_css)
        self.btn_screenshot.clicked.connect(self._screenshot)
        h.addWidget(self.btn_screenshot)

        self.btn_export = QPushButton("导出报告")
        self.btn_export.setStyleSheet(btn_css)
        self.btn_export.clicked.connect(self._export_report)
        h.addWidget(self.btn_export)

        h.addStretch()

        self.btn_about = QPushButton("关于")
        self.btn_about.setStyleSheet(btn_css)
        self.btn_about.clicked.connect(self._about)
        h.addWidget(self.btn_about)

        return header

    def _create_left_panel(self):
        """Left thumbnail panel with video list and layout controls."""
        panel = QWidget()
        panel.setObjectName("leftPanel")
        panel.setMinimumWidth(160)
        panel.setStyleSheet("QWidget#leftPanel { background: #1a1a2e; border-right: 1px solid #2a2a4a; }")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # header row
        hdr = QHBoxLayout()
        lbl = QLabel("视频列表")
        lbl.setStyleSheet("font-weight: bold; color: #8b8ba0; font-size: 13px; border: none;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        layout.addLayout(hdr)

        # video list
        self.source_list = QListWidget()
        self.source_list.setMinimumHeight(80)
        self.source_list.currentRowChanged.connect(self._on_video_selected)
        layout.addWidget(self.source_list, 1)

        # layout mode section
        sep = QLabel("布局模式")
        sep.setStyleSheet("color: #6b6b80; font-size: 11px; border: none; margin-top: 6px;")
        layout.addWidget(sep)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(4)
        mode_btn_css = (
            "QPushButton { background: #252540; color: #a0a0b0; border: 1px solid #333350; "
            "border-radius: 4px; font-size: 11px; padding: 4px 0px; }"
            "QPushButton:hover { background: #333358; color: #fff; }"
            "QPushButton:checked { background: #4f46e5; border-color: #4f46e5; color: #fff; }"
        )

        self.layout_btns = {}
        for mode_id, mode_text in [("1x1", "1×1"), ("2x2", "2×2"), ("1+2", "1+2")]:
            btn = QPushButton(mode_text)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setStyleSheet(mode_btn_css)
            btn.clicked.connect(lambda checked, m=mode_id: self._switch_layout(m))
            self.layout_btns[mode_id] = btn
            mode_layout.addWidget(btn)
        # default to 2x2
        self.layout_btns["2x2"].setChecked(True)
        self.layout_mode = "2x2"
        layout.addLayout(mode_layout)

        return panel

    def _create_right_panel(self):
        """Right control panel wrapped in a scroll area."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1a1a2e; }")
        scroll.setObjectName("rightPanel")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ---- control group ----
        ctrl_group = QGroupBox("控制面板")
        ctrl_group.setStyleSheet(self._group_style())
        ctrl_lay = QVBoxLayout(ctrl_group)
        ctrl_lay.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self.btn_start = QPushButton("开始检测")
        self.btn_start.setStyleSheet(self._btn_style("#10b981", "#059669"))
        self.btn_start.setMinimumHeight(38)
        self.btn_start.clicked.connect(self._start)
        row1.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(self._btn_style("#ef4444", "#dc2626"))
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.clicked.connect(self._stop)
        row1.addWidget(self.btn_stop)
        ctrl_lay.addLayout(row1)

        # ---- intervention group ----
        int_group = QGroupBox("干预操作")
        int_group.setStyleSheet(self._group_style())
        int_lay = QGridLayout(int_group)
        int_lay.setSpacing(6)

        self.btn_emergency = QPushButton("电梯急停")
        self.btn_emergency.setStyleSheet(self._btn_style("#ef4444", "#dc2626"))
        self.btn_emergency.clicked.connect(self._emergency_stop)
        int_lay.addWidget(self.btn_emergency, 0, 0)

        self.btn_call_staff = QPushButton("呼叫工作人员")
        self.btn_call_staff.setStyleSheet(self._btn_style("#f59e0b", "#d97706"))
        self.btn_call_staff.clicked.connect(self._call_staff)
        int_lay.addWidget(self.btn_call_staff, 0, 1)

        self.btn_dismiss = QPushButton("解除警报")
        self.btn_dismiss.setStyleSheet(self._btn_style("#6b7280", "#4b5563"))
        self.btn_dismiss.clicked.connect(self._dismiss_alert)
        int_lay.addWidget(self.btn_dismiss, 1, 0, 1, 2)

        ctrl_lay.addWidget(int_group)

        # ---- YOLO settings group ----
        yolo_group = QGroupBox("检测参数")
        yolo_group.setStyleSheet(self._group_style())
        yolo_lay = QGridLayout(yolo_group)
        yolo_lay.setSpacing(6)
        yolo_lay.setColumnStretch(1, 1)

        # row 0: YOLO confidence
        yolo_lay.addWidget(QLabel("检测置信度:"), 0, 0)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(50)
        self.conf_slider.setTickInterval(10)
        yolo_lay.addWidget(self.conf_slider, 0, 1)
        self.conf_label = QLabel("0.50")
        self.conf_label.setStyleSheet("color: #4f46e5; font-weight: bold; font-size: 12px; border: none;")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v / 100:.2f}"))
        yolo_lay.addWidget(self.conf_label, 0, 2)

        # row 1: skeleton keypoint confidence
        yolo_lay.addWidget(QLabel("骨骼置信度:"), 1, 0)
        self.skel_conf_slider = QSlider(Qt.Horizontal)
        self.skel_conf_slider.setRange(10, 80)
        self.skel_conf_slider.setValue(30)
        self.skel_conf_slider.setTickInterval(10)
        yolo_lay.addWidget(self.skel_conf_slider, 1, 1)
        self.skel_conf_label = QLabel("0.30")
        self.skel_conf_label.setStyleSheet("color: #4f46e5; font-weight: bold; font-size: 12px; border: none;")
        self.skel_conf_slider.valueChanged.connect(lambda v: self.skel_conf_label.setText(f"{v / 100:.2f}"))
        yolo_lay.addWidget(self.skel_conf_label, 1, 2)

        # row 2: frame skip
        yolo_lay.addWidget(QLabel("检测帧间隔:"), 2, 0)
        self.frame_skip_slider = QSlider(Qt.Horizontal)
        self.frame_skip_slider.setRange(1, 10)
        self.frame_skip_slider.setValue(1)
        self.frame_skip_slider.setTickInterval(1)
        yolo_lay.addWidget(self.frame_skip_slider, 2, 1)
        self.frame_skip_label = QLabel("1帧")
        self.frame_skip_label.setStyleSheet("color: #4f46e5; font-weight: bold; font-size: 12px; border: none;")
        self.frame_skip_slider.valueChanged.connect(lambda v: self.frame_skip_label.setText(f"{v}帧"))
        yolo_lay.addWidget(self.frame_skip_label, 2, 2)

        # row 3: alert cooldown
        yolo_lay.addWidget(QLabel("告警冷却:"), 3, 0)
        self.cooldown_slider = QSlider(Qt.Horizontal)
        self.cooldown_slider.setRange(1, 30)
        self.cooldown_slider.setValue(3)
        self.cooldown_slider.setTickInterval(5)
        yolo_lay.addWidget(self.cooldown_slider, 3, 1)
        self.cooldown_label = QLabel("3秒")
        self.cooldown_label.setStyleSheet("color: #4f46e5; font-weight: bold; font-size: 12px; border: none;")
        self.cooldown_slider.valueChanged.connect(lambda v: self.cooldown_label.setText(f"{v}秒"))
        yolo_lay.addWidget(self.cooldown_label, 3, 2)

        # row 4: recall rate / detection sensitivity
        yolo_lay.addWidget(QLabel("扶梯方向:"), 4, 0)
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(["上行", "下行"])
        self.dir_combo.setMinimumHeight(30)
        yolo_lay.addWidget(self.dir_combo, 4, 1, 1, 2)

        ctrl_lay.addWidget(yolo_group)

        # ---- stats group ----
        stats_group = QGroupBox("行为统计")
        stats_group.setStyleSheet(self._group_style())
        stats_lay = QGridLayout(stats_group)
        stats_lay.setSpacing(4)
        self.stats_labels = {}
        for idx, (key, cn, color) in enumerate(
            [
                ("normal", "正常", "#10b981"),
                ("running", "奔跑", "#ef4444"),
                ("reverse", "逆行", "#ef4444"),
                ("falling", "摔倒", "#ef4444"),
                ("loitering", "滞留", "#ef4444"),
                ("sitting", "坐下", "#f59e0b"),
            ]
        ):
            row, col = divmod(idx, 2)
            lbl = QLabel(f"{cn}: 0")
            lbl.setStyleSheet(
                f"color: {color}; font-size: 13px; font-weight: bold; border: none; background: transparent;"
            )
            self.stats_labels[key] = lbl
            stats_lay.addWidget(lbl, row, col)
        ctrl_lay.addWidget(stats_group)

        ctrl_lay.addStretch()

        layout.addWidget(ctrl_group)
        scroll.setWidget(content)
        return scroll

    def _create_bottom_panel(self):
        """Bottom alert table panel."""
        panel = QWidget()
        panel.setObjectName("bottomPanel")
        panel.setMinimumHeight(140)
        panel.setMaximumHeight(360)
        panel.setStyleSheet("QWidget#bottomPanel { background: #1a1a2e; border-top: 1px solid #2a2a4a; }")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # header
        hdr = QHBoxLayout()
        lbl = QLabel("实时告警记录")
        lbl.setStyleSheet("font-weight: bold; color: #8b8ba0; font-size: 13px; border: none;")
        hdr.addWidget(lbl)
        hdr.addStretch()

        self.alert_count_label = QLabel("共 0 条")
        self.alert_count_label.setStyleSheet("color: #6b6b80; font-size: 11px; border: none;")
        hdr.addWidget(self.alert_count_label)

        clear_btn = QPushButton("清空记录")
        clear_btn.setFixedHeight(24)
        clear_btn.clicked.connect(lambda: self._clear_alerts())
        clear_btn.setStyleSheet(
            "QPushButton { background: #252540; color: #8b8ba0; border: none; "
            "border-radius: 4px; font-size: 11px; padding: 2px 12px; }"
            "QPushButton:hover { background: #35355a; color: #fff; }"
        )
        hdr.addWidget(clear_btn)
        layout.addLayout(hdr)

        self.alert_table = QTableWidget(0, 5)
        self.alert_table.setHorizontalHeaderLabels(["时间", "行人ID", "行为", "置信度", "视频源"])
        self.alert_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.alert_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.alert_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.alert_table.verticalHeader().setVisible(False)
        self.alert_table.setAlternatingRowColors(True)
        self.alert_table.setStyleSheet("QTableWidget { alternate-background-color: #14142b; }")
        layout.addWidget(self.alert_table)

        return panel

    def _setup_video_grid(self, count=None, layout_mode=None):
        """Setup/reset the video grid with given count and layout mode."""
        if count is None:
            count = max(len(self.video_sources), 1)
        if layout_mode is None:
            layout_mode = self.layout_mode

        # clear old widgets
        for lbl in self.video_labels:
            self.video_grid.removeWidget(lbl)
            lbl.deleteLater()
        self.video_labels.clear()

        # determine grid structure
        if layout_mode == "1x1":
            cols, _rows = 1, 1
            show_count = 1
            active = self.active_video_idx
            for i in range(show_count):
                vid_idx = (active + i) % max(count, 1)
                vl = VideoLabel(vid_idx)
                self.video_labels.append(vl)
                self.video_grid.addWidget(vl, 0, 0)
        elif layout_mode == "1+2":
            cols, _rows = 2, 2
            show_count = min(count, 3)
            active = min(self.active_video_idx, count - 1)
            # video 0 (active) spans 2 rows
            vl0 = VideoLabel(active)
            self.video_labels.append(vl0)
            self.video_grid.addWidget(vl0, 0, 0, 2, 1)
            # remaining videos in right column
            others = [i for i in range(count) if i != active]
            for j, vid_idx in enumerate(others[:2]):
                vl = VideoLabel(vid_idx)
                self.video_labels.append(vl)
                self.video_grid.addWidget(vl, j, 1)
            self.video_grid.setColumnStretch(0, 2)
            self.video_grid.setColumnStretch(1, 1)
        else:  # '2x2' or 'auto'
            if count <= 2:
                cols, _rows = count, 1
            else:
                cols, _rows = 2, (count + 1) // 2
            show_count = min(count, 4)
            for i in range(show_count):
                vl = VideoLabel(i)
                self.video_labels.append(vl)
                row, col = divmod(i, cols)
                self.video_grid.addWidget(vl, row, col)

        # update labels with video source names
        for i, (name, path, is_cam) in enumerate(self.video_sources):
            if i < len(self.video_labels) and self.video_labels[i].video_idx == i:
                self.video_labels[i].setText(f"{name}\n待开始检测")
            elif i < len(self.video_labels):
                idx = self.video_labels[i].video_idx
                if idx < len(self.video_sources):
                    self.video_labels[i].setText(f"{self.video_sources[idx][0]}\n待开始检测")

    def _switch_layout(self, mode):
        """Switch video grid layout mode."""
        self.layout_mode = mode
        for m, btn in self.layout_btns.items():
            btn.setChecked(m == mode)
        self._setup_video_grid(layout_mode=mode)

    def _toggle_left_panel(self):
        """Toggle left panel visibility."""
        self.left_panel_visible = not self.left_panel_visible
        self.left_panel_widget.setVisible(self.left_panel_visible)
        self.btn_toggle_left.setText("◀" if self.left_panel_visible else "▶")

    def _on_video_selected(self, row):
        """Handle video selection from list."""
        if row >= 0 and row < len(self.video_sources):
            self.active_video_idx = row
            if self.layout_mode in ("1x1", "1+2"):
                self._setup_video_grid(layout_mode=self.layout_mode)

    def _clear_alerts(self):
        self.alert_table.setRowCount(0)
        self.alert_count_label.setText("共 0 条")

    def _add_video_source(self):
        if len(self.video_sources) >= 4:
            QMessageBox.warning(self, "提示", "最多支持4路视频")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*.*)"
        )
        if not path:
            return
        self.video_sources.append((os.path.basename(path), path, False))
        self._setup_video_grid()
        self._update_source_list()

    def _remove_video_source(self):
        if self.video_sources:
            self.video_sources.pop()
            if self.active_video_idx >= len(self.video_sources):
                self.active_video_idx = max(0, len(self.video_sources) - 1)
            self._setup_video_grid()
            self._update_source_list()

    def _update_source_list(self):
        self.source_list.blockSignals(True)
        self.source_list.clear()
        for name, path, is_cam in self.video_sources:
            prefix = "[CAM]" if is_cam else "[VID]"
            self.source_list.addItem(f"{prefix} {name}")
        self.source_list.blockSignals(False)
        if self.video_sources:
            self.source_list.setCurrentRow(self.active_video_idx)

    def _toggle_roi_mode(self, checked):
        self.roi_mode = checked
        for vl in self.video_labels:
            vl.set_roi_mode(checked)
        self.btn_roi_mode.setChecked(checked)
        self.status_label.setText("ROI模式: 拖拽绘制检测区域" if checked else "ROI模式已关闭")

    def _clear_all_roi(self):
        for vl in self.video_labels:
            vl.clear_roi()
        self.status_label.setText("所有ROI区域已清除")

    def _init_statusbar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("color: #6b6b80; font-size: 14px; border: none; padding: 0 4px;")
        self.status_bar.addWidget(self.status_indicator)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("padding: 0 4px;")
        self.status_bar.addWidget(self.status_label, 1)

        self.video_count_label = QLabel("视频: 0路")
        self.status_bar.addPermanentWidget(self.video_count_label)

        self.mode_label = QLabel("模式: 待机")
        self.status_bar.addPermanentWidget(self.mode_label)

        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_short = gpu_name.replace("NVIDIA GeForce ", "") if "NVIDIA" in gpu_name else gpu_name
            gpu_text = f"GPU: {gpu_short}"
        else:
            gpu_text = "GPU: CPU"
        self.gpu_label = QLabel(gpu_text)
        self.status_bar.addPermanentWidget(self.gpu_label)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-weight: bold; color: #4f46e5; border: none; padding: 0 8px;")
        self.status_bar.addPermanentWidget(self.fps_label)

    def _group_style(self):
        return (
            "QGroupBox { font-weight: bold; font-size: 13px; color: #c0c0d0; "
            "border: 1px solid #2a2a4a; border-radius: 8px; "
            "margin-top: 14px; padding: 16px 10px 10px 10px; background: #1a1a2e; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 14px; "
            "padding: 0 8px; color: #8b8ba0; }"
        )

    def _btn_style(self, bg, hover):
        return (
            f"QPushButton {{ background: {bg}; color: white; border: none; "
            f"padding: 10px; border-radius: 6px; font-size: 13px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {hover}; }}"
            f"QPushButton:disabled {{ background: #3b3b52; color: #606070; }}"
        )

    # ========== 控制逻辑 ==========
    def _start(self):
        if not self.video_sources:
            QMessageBox.warning(self, "提示", "请先添加至少一个视频源")
            return

        self.status_label.setText("加载模型中...")
        self.status_label.setStyleSheet("color: #f59e0b; font-weight: bold; padding: 0 4px;")
        self.status_indicator.setStyleSheet("color: #f59e0b; font-size: 14px; border: none; padding: 0 4px;")
        QApplication.processEvents()

        print("[GUI] 加载 YOLO...")
        self.pose_model = YOLO("yolov8n-pose.pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.escalator_direction = "up" if self.dir_combo.currentIndex() == 0 else "down"
        print(f"[GUI] 加载 ST-GCN (device={device})...")
        self.action_recognizer = ActionRecognizer(
            weight_path="st_gcn.kinetics.pt", escalator_direction=self.escalator_direction, device=device
        )
        self.conf_threshold = self.conf_slider.value() / 100.0
        self.skeleton_conf = self.skel_conf_slider.value() / 100.0
        self.frame_skip = self.frame_skip_slider.value()
        self.alert_cooldown = self.cooldown_slider.value()

        num = len(self.video_sources)
        self.caps = []
        self.names = []
        self.rois = {}
        self.track_hists = []
        self.track_last = []
        self.counters = []
        self.fps_deques = []
        self.prev_ts = []
        self._alert_cooldown_times = {}
        self._frame_counter = 0

        for i, (name, src, is_cam) in enumerate(self.video_sources):
            cap_src = 0 if is_cam else src
            print(f"[GUI] 打开视频: {name}")
            cap = cv2.VideoCapture(cap_src)
            self.caps.append(cap if cap.isOpened() else None)
            self.names.append(name)
            self.track_hists.append(defaultdict(lambda: deque(maxlen=self.seq_length * 2)))
            self.track_last.append({})
            self.counters.append(defaultdict(int))
            self.fps_deques.append(deque(maxlen=30))
            self.prev_ts.append(time.time())
        # collect ROIs from video labels (labels may be in non-sequential order)
        for vl in self.video_labels:
            if vl.roi is not None:
                self.rois[vl.video_idx] = vl.roi

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
        self.btn_screenshot.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.status_label.setText("运行中")
        self.status_label.setStyleSheet("color: #10b981; font-weight: bold; padding: 0 4px;")
        self.status_indicator.setStyleSheet("color: #10b981; font-size: 14px; border: none; padding: 0 4px;")
        self.mode_label.setText(f"模式: 检测{'上行' if self.escalator_direction == 'up' else '下行'}")
        self.video_count_label.setText(f"视频: {num}路")
        self.alert_table.setRowCount(0)
        self.alert_count_label.setText("共 0 条")

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
        if not self.running or not self.caps:
            return
        if getattr(self, "_busy", False):
            return
        self._busy = True

        num = len(self.caps)
        if not any(c and c.isOpened() for c in self.caps):
            self._stop()
            self.status_label.setText("处理完成")
            self.status_label.setStyleSheet("padding: 0 4px;")
            self._busy = False
            return

        # frame skip for performance
        self._frame_counter += 1
        if self._frame_counter % self.frame_skip != 0:
            # still read the frame to advance video position
            idx = self.rr % num
            self.rr += 1
            cap = self.caps[idx]
            if cap and cap.isOpened():
                cap.read()  # advance but discard
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

        results = self.pose_model.track(
            frame, persist=True, tracker="bytetrack.yaml", conf=self.conf_threshold, verbose=False
        )
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
                    all_persons.append({"track_id": tid, "bbox": b, "kpts": kpts_arr[j], "center": ctr})

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
                others = [p for p in all_persons if p["track_id"] != tid]
                self.track_hists[idx][tid].append(kpts.copy())

                if tid not in self.track_last[idx]:
                    self.track_last[idx][tid] = "normal"

                action_en, action_cn, conf = self.action_recognizer.predict(
                    tid, list(self.track_hists[idx][tid]), bbox, (h, w), all_persons=others
                )
                self.track_last[idx][tid] = action_en
                self.counters[idx][action_cn] += 1

                if action_en in DANGER_ACTIONS or action_en in CAUTION_ACTIONS:
                    now_ts = time.time()
                    ck = (idx, tid, action_en)
                    last = self._alert_cooldown_times.get(ck, 0)
                    if now_ts - last >= self.alert_cooldown:
                        self._alert_cooldown_times[ck] = now_ts
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
                label = f"ID:{tid} {action_cn}"
                annotated = cv2_draw_chinese(annotated, label, (x1, max(y1 - 28, 0)), font_size=18, color=color)

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
            (10, 8),
            font_size=18,
            color=(255, 255, 255),
        )

        now = time.time()
        dt = max(now - self.prev_ts[idx], 0.001)
        self.fps_deques[idx].append(1.0 / dt)
        self.prev_ts[idx] = now
        afps = sum(self.fps_deques[idx]) / len(self.fps_deques[idx])
        self.fps_label.setText(f"FPS: {afps:.1f}")

        # find the correct video label by video index
        for vl in self.video_labels:
            if vl.video_idx == idx:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                hh, ww, ch = rgb.shape
                qimg = QImage(rgb.copy().data, ww, hh, ch * ww, QImage.Format_RGB888)
                vl._orig_size = (ww, hh)
                scaled = qimg.scaled(vl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                vl.setPixmap(QPixmap.fromImage(scaled))
                break

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
        self.alert_table.rowCount()
        self.alert_table.insertRow(0)

        items = [
            QTableWidgetItem(timestamp),
            QTableWidgetItem(str(track_id)),
            QTableWidgetItem(action_cn),
            QTableWidgetItem(f"{confidence:.2f}"),
            QTableWidgetItem(self.names[video_idx] if video_idx < len(self.names) else "-"),
        ]
        for c, item in enumerate(items):
            self.alert_table.setItem(0, c, item)

        if action_cn in ["奔跑", "逆行", "摔倒", "滞留"]:
            for c in range(5):
                self.alert_table.item(0, c).setForeground(QColor("#ef4444"))
            self._start_alarm()
        elif action_cn == "坐下":
            for c in range(5):
                self.alert_table.item(0, c).setForeground(QColor("#f59e0b"))

        while self.alert_table.rowCount() > 100:
            self.alert_table.removeRow(self.alert_table.rowCount() - 1)
        self.alert_count_label.setText(f"共 {self.alert_table.rowCount()} 条")

    def _start_alarm(self):
        if self.emergency_stop_active:
            return
        if hasattr(self, "_alarm_thread") and self._alarm_thread and self._alarm_thread.is_alive():
            return
        self._alarm_stop = threading.Event()
        self._alarm_thread = threading.Thread(target=play_alarm_sound, args=(self._alarm_stop,), daemon=True)
        self._alarm_thread.start()

    def _stop_alarm(self):
        if hasattr(self, "_alarm_stop"):
            self._alarm_stop.set()

    def _call_staff(self):
        self._stop_alarm()
        self.status_label.setText("[!] 已呼叫工作人员 - 请等待响应")
        self.status_label.setStyleSheet("color: #f59e0b; font-weight: bold; padding: 0 4px;")
        self.status_indicator.setStyleSheet("color: #f59e0b; font-size: 14px; border: none; padding: 0 4px;")

    def _emergency_stop(self):
        self._stop_alarm()
        self.emergency_stop_active = True
        self.status_label.setText("[!!!] 电梯已急停 - 检测继续运行中")
        self.status_label.setStyleSheet("color: #ef4444; font-weight: bold; padding: 0 4px;")
        self.status_indicator.setStyleSheet("color: #ef4444; font-size: 14px; border: none; padding: 0 4px;")

    def _dismiss_alert(self):
        self._stop_alarm()
        self.emergency_stop_active = False
        self.status_label.setText("就绪 - 警报已解除")
        self.status_label.setStyleSheet("color: #10b981; font-weight: bold; padding: 0 4px;")
        self.status_indicator.setStyleSheet("color: #10b981; font-size: 14px; border: none; padding: 0 4px;")

    def _reset_ui_state(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_add_video.setEnabled(True)
        self.btn_remove_video.setEnabled(True)
        self.btn_roi_mode.setEnabled(True)
        self.btn_clear_roi.setEnabled(True)
        self.btn_screenshot.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.fps_label.setText("FPS: --")
        self.status_label.setText("就绪")
        self.status_label.setStyleSheet("padding: 0 4px;")
        self.status_indicator.setStyleSheet("color: #6b6b80; font-size: 14px; border: none; padding: 0 4px;")
        self.mode_label.setText("模式: 待机")
        self.video_count_label.setText(f"视频: {len(self.video_sources)}路")
        self.emergency_stop_active = False

    def _screenshot(self):
        for i, vl in enumerate(self.video_labels):
            if vl.pixmap():
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                vl.pixmap().save(f"screenshot_ch{vl.video_idx}_{ts}.png")
        self.status_label.setText("截图已保存")
        self.status_label.setStyleSheet("padding: 0 4px;")

    def _export_report(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出报告", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "文本文件 (*.txt)"
        )
        if not path:
            return
        try:
            from database import get_alert_stats, get_recent_alerts

            with open(path, "w", encoding="utf-8") as f:
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
            self.status_label.setStyleSheet("padding: 0 4px;")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")

    def _about(self):
        QMessageBox.about(
            self,
            "关于",
            "自动扶梯行人安全检测系统 v2.0\n\n"
            "技术栈: YOLOv8-Pose + ByteTrack + ST-GCN\n"
            "使用规则判别 + 深度学习混合方案\n\n"
            "功能: 多路视频监控 | ROI区域检测 | 声音报警 | 干预控制",
        )

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
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    app.setFont(QFont("Microsoft YaHei", 10))

    login = LoginWindow()
    if login.exec_() != QDialog.Accepted:
        sys.exit(0)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
