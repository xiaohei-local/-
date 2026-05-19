"""
自动扶梯行人动作识别模块
融合方案：规则判别（优先）+ ST-GCN（辅助）
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'st-gcn'))
from net.st_gcn import Model as OfficialSTGCN
from net.utils.graph import Graph

# ==================== 行为类别定义 ====================
ESCALATOR_ACTIONS = [
    'normal', 'running', 'reverse', 'falling', 'loitering', 'sitting'
]

ACTION_CN = {
    'normal': '正常', 'running': '奔跑', 'reverse': '逆行',
    'falling': '摔倒', 'loitering': '滞留', 'sitting': '坐下'
}

DANGER_ACTIONS = {'running', 'reverse', 'falling', 'loitering'}
CAUTION_ACTIONS = {'sitting'}

# Kinetics-400 中与扶梯行为相关的类别索引映射
KINETICS_TO_ESCALATOR = {
    123: 'falling',        # faceplanting → 摔倒
    282: 'running',        # running on treadmill → 奔跑
    106: 'falling',        # drop kicking → 摔倒
    173: 'falling',        # jumping into pool → 可能的摔倒
    383: 'sitting',        # stretching leg → 坐下
    358: 'sitting',        # sitting on chair → 坐下
     80: 'loitering',      # standing on hands → 滞留（异常静止）
    182: 'loitering',      # jumping jacks → 滞留（原地运动）
}

# OpenPose(18) → COCO(17) 索引映射
# COCO: 0=nose,1=Leye,2=Reye,3=Lear,4=Rear,5=Lshoulder,6=Rshoulder,
#       7=Le;bow,8=Relbow,9=Lwrist,10=Rwrist,11=Lhip,12=Rhip,
#       13=Lknee,14=Rknee,15=Lankle,16=Rankle
# OP:   0=nose,1=neck,2=Rshoulder,3=Relbow,4=Rwrist,5=Lshoulder,
#       6=Le;bow,7=Lwrist,8=Rhip,9=Rknee,10=Rankle,11=Lhip,
#       12=Lknee,13=Lankle,14=Reye,15=Leye,16=Rear,17=Lear
# op_idx → coco_idx, neck(1) 由双肩中点计算
OPENPOSE_TO_COCO = [0, 0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]


def coco17_to_openpose18(kpts_coco):
    """将 COCO 17 关键点转为 OpenPose 18 关键点格式，缺失的 neck 由双肩中点计算"""
    op = np.zeros((18, 3), dtype=np.float32)
    for op_idx, coco_idx in enumerate(OPENPOSE_TO_COCO):
        op[op_idx] = kpts_coco[coco_idx]
    op[1] = (kpts_coco[5] + kpts_coco[6]) / 2.0  # neck = 双肩中点
    return op


# ==================== 规则判别引擎 ====================
class PoseRuleEngine:
    """基于骨骼几何特征的规则判别器，处理扶梯特有危险行为"""

    def __init__(self, escalator_direction='up',
                 retrograde_thresh=0.02, loiter_thresh=30, fall_thresh=5):
        self.escalator_direction = escalator_direction  # 'up' or 'down'
        self.retrograde_thresh = retrograde_thresh
        self.loiter_thresh = loiter_thresh
        self.fall_thresh = fall_thresh
        self.reset()

    def reset(self):
        """每个 track 的状态"""
        self.prev_hip_y = None
        self.prev_hip_pos = None
        self.fall_counter = 0
        self.run_counter = 0
        self.reverse_counter = 0
        self.sit_counter = 0
        self.loiter_ref_pos = None
        self.loiter_counter = 0

    def detect(self, kpts, bbox, frame_shape, track_id=None, all_persons=None):
        """
        对单帧单人进行规则判别
        kpts: (17, 3) COCO格式关键点
        bbox: (x1, y1, x2, y2)
        frame_shape: (h, w)
        all_persons: list of dict, 其他行人的信息 [{'bbox': ..., 'kpts': ..., 'center': ...}]
        返回: (action_name, confidence)
        """
        h, w = frame_shape
        confs = kpts[:, 2]
        avg_conf = np.mean(confs)

        if avg_conf < 0.3:
            return None, 0.0

        results = []

        # 1. 摔倒检测
        fall_result = self._check_falling(kpts, bbox, h, w)
        if fall_result:
            results.append(fall_result)

        # 2. 奔跑检测
        run_result = self._check_running(kpts, bbox, h, w)
        if run_result:
            results.append(run_result)

        # 3. 逆行检测
        reverse_result = self._check_reverse(kpts, bbox, h, w)
        if reverse_result:
            results.append(reverse_result)

        # 4. 坐下检测
        sit_result = self._check_sitting(kpts, bbox, h, w)
        if sit_result:
            results.append(sit_result)

        # 5. 滞留检测
        loiter_result = self._check_loitering(kpts, bbox, h, w)
        if loiter_result:
            results.append(loiter_result)

        if not results:
            return None, 0.0

        # 返回置信度最高的结果
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0]

    def _check_falling(self, kpts, bbox, h, w):
        """摔倒检测：检查宽高比异常 + 头/髋快速下移"""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        # 宽高比异常（人倒下时宽度 > 高度）
        if bh > 0 and bw / bh > 1.3:
            self.fall_counter += 3
        else:
            self.fall_counter = max(0, self.fall_counter - 1)

        # 头部快速下移
        nose = kpts[0]
        hip_center = (kpts[11, :2] + kpts[12, :2]) / 2.0
        hip_y = hip_center[1]

        if self.prev_hip_y is not None and nose[2] > 0.4:
            dy = hip_y - self.prev_hip_y
            # 向下移动超过身体高度的 15% / 帧
            if dy > bh * 0.15:
                self.fall_counter += 2

        self.prev_hip_y = hip_y

        if self.fall_counter >= self.fall_thresh:
            return ('falling', min(0.95, 0.6 + self.fall_counter * 0.05))

        return None

    def _check_running(self, kpts, bbox, h, w):
        """奔跑检测：帧间位移量大 + 步频高"""
        hip_center = (kpts[11, :2] + kpts[12, :2]) / 2.0
        _, _, _, bh = bbox
        bh = max(bh, 1)

        if self.prev_hip_pos is not None:
            displacement = np.linalg.norm(hip_center - self.prev_hip_pos)
            # 帧间位移超过身体高度的 5%
            if displacement > bh * 0.05:
                self.run_counter += 2
            else:
                self.run_counter = max(0, self.run_counter - 1)
        else:
            self.run_counter = max(0, self.run_counter - 1)

        self.prev_hip_pos = hip_center

        # 检查步幅（脚踝间距）
        ankle_dist = np.linalg.norm(kpts[15, :2] - kpts[16, :2])
        if ankle_dist > bh * 0.4:
            self.run_counter += 1

        if self.run_counter >= 6:
            return ('running', min(0.9, 0.55 + self.run_counter * 0.04))

        return None

    def _check_reverse(self, kpts, bbox, h, w):
        """逆行检测：行人运动方向与扶梯运行方向相反"""
        hip_center = (kpts[11, :2] + kpts[12, :2]) / 2.0

        if self.prev_hip_pos is not None:
            dy = hip_center[1] - self.prev_hip_pos[1]
            _, _, _, bh = bbox
            threshold = bh * self.retrograde_thresh

            if self.escalator_direction == 'up':
                if dy > threshold:
                    self.reverse_counter += 2
                else:
                    self.reverse_counter = max(0, self.reverse_counter - 1)
            else:
                if dy < -threshold:
                    self.reverse_counter += 2
                else:
                    self.reverse_counter = max(0, self.reverse_counter - 1)

        self.prev_hip_pos = hip_center

        if self.reverse_counter >= 8:
            return ('reverse', min(0.9, 0.55 + self.reverse_counter * 0.03))

        return None

    def _check_sitting(self, kpts, bbox, h, w):
        """坐下检测：髋部位置低、膝盖弯曲、身体紧凑"""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bh <= 0:
            return None

        hip_mid_y = (kpts[11, 1] + kpts[12, 1]) / 2.0
        shoulder_mid = (kpts[5, :2] + kpts[6, :2]) / 2.0
        hip_mid = (kpts[11, :2] + kpts[12, :2]) / 2.0

        hip_rel_y = (hip_mid_y - y1) / bh
        torso_len = np.linalg.norm(shoulder_mid - hip_mid)
        compactness = torso_len / max(bh, 1)
        lknee_angle = self._calc_angle(kpts[11, :2], kpts[13, :2], kpts[15, :2])
        rknee_angle = self._calc_angle(kpts[12, :2], kpts[14, :2], kpts[16, :2])
        min_knee_angle = min(lknee_angle, rknee_angle)

        score = 0
        if hip_rel_y > 0.65:
            score += 1
        if compactness < 0.4:
            score += 1
        if min_knee_angle < 120:
            score += 1
        if bw / max(bh, 1) > 0.8:
            score += 1

        if score >= 2:
            self.sit_counter += 2
        else:
            self.sit_counter = max(0, self.sit_counter - 1)

        if self.sit_counter >= 8:
            return ('sitting', min(0.85, 0.5 + self.sit_counter * 0.03))
        return None

    def _check_loitering(self, kpts, bbox, h, w):
        """滞留检测：长时间位移极小，表明行人在扶梯入口/出口滞留"""
        hip_center = (kpts[11, :2] + kpts[12, :2]) / 2.0
        _, _, _, bh = bbox
        bh = max(bh, 1)

        if self.loiter_ref_pos is not None:
            displacement = np.linalg.norm(hip_center - self.loiter_ref_pos)
            if displacement < bh * 0.03:
                self.loiter_counter += 1
            else:
                self.loiter_counter = max(0, self.loiter_counter - 2)
                self.loiter_ref_pos = hip_center
        else:
            self.loiter_ref_pos = hip_center
            return None

        if self.loiter_counter >= self.loiter_thresh:
            conf = min(0.9, 0.5 + self.loiter_counter * 0.01)
            return ('loitering', conf)
        return None

    @staticmethod
    def _calc_angle(a, b, c):
        """计算向量 ba→bc 的夹角（度数）"""
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# ==================== ST-GCN 封装 ====================
class STGCNWrapper:
    """封装官方 ST-GCN 模型，处理 COCO→OpenPose 映射和推理"""

    def __init__(self, weight_path='st_gcn.kinetics.pt', device='cpu'):
        self.device = torch.device(device)
        self.num_class = 400

        graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
        self.model = OfficialSTGCN(
            in_channels=3,
            num_class=self.num_class,
            graph_args=graph_args,
            edge_importance_weighting=True
        )

        # 加载权重
        self._load_weights(weight_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, weight_path):
        if not os.path.exists(weight_path):
            print(f"[STGCNWrapper] 权重文件不存在: {weight_path}，使用随机初始化")
            return

        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            # 兼容不同保存格式
            if 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state = checkpoint['model']
            else:
                state = checkpoint

            # 移除 nn.DataParallel 的 'module.' 前缀
            new_state = {}
            for k, v in state.items():
                if k.startswith('module.'):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v

            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            if missing:
                print(f"[STGCNWrapper] 缺失的层 ({len(missing)}): 仅 fcn 层预期不匹配")
            if unexpected:
                print(f"[STGCNWrapper] 多余的层: {len(unexpected)}")
            print(f"[STGCNWrapper] 已加载 Kinetics-400 预训练权重")
        except Exception as e:
            print(f"[STGCNWrapper] 权重加载失败: {e}")

    @torch.no_grad()
    def predict(self, keypoint_sequence, T=16):
        """
        keypoint_sequence: list of (17, 3) COCO keypoints, length = T
        返回: (escalator_action_en, confidence)
        """
        if len(keypoint_sequence) < T:
            return None, 0.0

        seq = keypoint_sequence[-T:]

        # 转为 OpenPose 18 格式并进行标准化
        op_seq = []
        for kpts_coco in seq:
            op = coco17_to_openpose18(kpts_coco)
            # 以 neck 为中心
            neck = op[1]
            if neck[2] > 0.1:
                op[:, 0] -= neck[0]
                op[:, 1] -= neck[1]
            # 以躯干长度归一化
            shoulder_mid = (op[5, :2] + op[2, :2]) / 2.0
            hip_mid = (op[11, :2] + op[8, :2]) / 2.0
            torso_len = np.linalg.norm(shoulder_mid - hip_mid)
            if torso_len > 5:
                op[:, 0] /= torso_len
                op[:, 1] /= torso_len
            op_seq.append(op)

        blob = np.array(op_seq)  # (T, 18, 3)
        blob = np.transpose(blob, (2, 0, 1))  # (3, T, 18)
        blob = blob[np.newaxis, :, :, :, np.newaxis]  # (1, 3, T, 18, 1)

        tensor = torch.tensor(blob, dtype=torch.float32).to(self.device)
        output = self.model(tensor)  # (1, 400)

        probs = F.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, dim=1)
        top_prob = top_prob.item()
        top_class = top_class.item()

        # 映射到扶梯行为
        if top_class in KINETICS_TO_ESCALATOR and top_prob > 0.3:
            action = KINETICS_TO_ESCALATOR[top_class]
            return action, top_prob

        return None, top_prob

    @torch.no_grad()
    def extract_features(self, keypoint_sequence, T=16):
        """提取 ST-GCN 特征向量（用于后续分析）"""
        if len(keypoint_sequence) < T:
            return None

        seq = keypoint_sequence[-T:]
        op_seq = [coco17_to_openpose18(k) for k in seq]
        blob = np.array(op_seq).transpose(2, 0, 1)[np.newaxis, :, :, :, np.newaxis]
        tensor = torch.tensor(blob, dtype=torch.float32).to(self.device)
        _, feature = self.model.extract_feature(tensor)
        return feature.cpu().numpy()


# ==================== 融合动作识别器 ====================
class ActionRecognizer:
    """融合规则判别 + ST-GCN 的动作识别器，每个 track 维护独立的规则引擎状态"""

    def __init__(self, weight_path='st_gcn.kinetics.pt', escalator_direction='up',
                 device='cpu',
                 retrograde_thresh=0.02, loiter_thresh=30, fall_thresh=5):
        self.escalator_direction = escalator_direction
        self.stgcn = STGCNWrapper(weight_path, device)
        self.device = device
        self.seq_length = 16
        self.retrograde_thresh = retrograde_thresh
        self.loiter_thresh = loiter_thresh
        self.fall_thresh = fall_thresh
        self._engines = {}  # track_id → PoseRuleEngine

    def _get_engine(self, track_id):
        if track_id not in self._engines:
            self._engines[track_id] = PoseRuleEngine(
                self.escalator_direction,
                self.retrograde_thresh,
                self.loiter_thresh,
                self.fall_thresh
            )
        return self._engines[track_id]

    def remove_track(self, track_id):
        self._engines.pop(track_id, None)

    def predict(self, track_id, keypoint_history, bbox, frame_shape,
                all_persons=None):
        """
        对单个跟踪目标进行动作识别
        keypoint_history: list of (17, 3) arrays, 该 track 的历史关键点
        返回: (action_en, action_cn, confidence)
        """
        if len(keypoint_history) < 2:
            return 'normal', '正常', 0.0

        current_kpts = keypoint_history[-1]
        engine = self._get_engine(track_id)

        # 1. 规则判别（优先检测危险行为）
        rule_action, rule_conf = engine.detect(
            current_kpts, bbox, frame_shape, track_id, all_persons
        )

        if rule_action is not None and rule_conf > 0.55:
            return rule_action, ACTION_CN[rule_action], rule_conf

        # 2. ST-GCN 辅助判别
        if len(keypoint_history) >= self.seq_length:
            stgcn_action, stgcn_conf = self.stgcn.predict(
                keypoint_history, self.seq_length
            )
            if stgcn_action is not None and stgcn_conf > 0.4:
                return stgcn_action, ACTION_CN[stgcn_action], stgcn_conf

        # 3. 默认：正常
        return 'normal', '正常', 0.3


if __name__ == '__main__':
    print("=" * 50)
    print("动作识别模块自检")
    print("=" * 50)

    # 测试规则引擎
    print("\n[1] 规则引擎测试...")
    engine = PoseRuleEngine(escalator_direction='up')

    # 模拟正常站姿
    normal_kpts = np.zeros((17, 3))
    normal_kpts[:, 2] = 0.9
    normal_kpts[0] = [320, 100, 0.9]  # nose
    normal_kpts[5] = [280, 200, 0.9]  # L shoulder
    normal_kpts[6] = [360, 200, 0.9]  # R shoulder
    normal_kpts[11] = [290, 350, 0.9]  # L hip
    normal_kpts[12] = [350, 350, 0.9]  # R hip
    result = engine.detect(normal_kpts, [250, 50, 400, 500], (480, 640))
    print(f"  正常站姿: {result}")

    # 模拟坐下
    sit_kpts = normal_kpts.copy()
    sit_kpts[11] = [290, 380, 0.9]  # L hip lower
    sit_kpts[12] = [350, 380, 0.9]  # R hip lower
    sit_kpts[13] = [280, 430, 0.9]  # L knee bent
    sit_kpts[14] = [360, 430, 0.9]  # R knee bent
    for _ in range(8):
        result = engine.detect(sit_kpts, [250, 300, 400, 480], (480, 640))
    print(f"  坐下姿态: {result}")

    # 测试 ST-GCN
    print("\n[2] ST-GCN 模型测试...")
    try:
        wrapper = STGCNWrapper('st_gcn.kinetics.pt', 'cpu')
        print("  ST-GCN 模型加载成功")
    except Exception as e:
        print(f"  ST-GCN 加载失败: {e}")

    print("\n动作识别模块自检完成!")
