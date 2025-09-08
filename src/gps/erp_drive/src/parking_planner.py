#!/usr/bin/env python3
import math
import numpy as np
import rospy

from geometry_msgs.msg import Pose
from lidar_object_detection.msg import ObjectInfo
from std_msgs.msg import Int16
from tf.transformations import euler_from_quaternion

# 주차 경로/요소는 이후 단계에서 사용할 예정 (잔존 import 유지)
from erp_drive.msg import PathReference, PathReferenceElement


class ParkingPlanner:
    def __init__(self, rviz_offset_x: float, rviz_offset_y: float):
        # === RViz 보조(그대로) ===
        self.rviz_offset_x = float(rviz_offset_x)
        self.rviz_offset_y = float(rviz_offset_y)

        # === 기본 파라미터 ===
        self.frame_id = "map"
        self.ROI_radius = 1.5  # 차량 기준 ROI (m)

        # === 트래킹/안정화 파라미터 ===
        self.match_gate = 0.25     # 트랙 매칭 허용 거리 (m)
        self.stable_k   = 10       # count 임계값 (>= 이면 안정 좌표 편입)
        self.decay_miss = 10       # 미관측 허용 프레임(지나면 트랙 삭제)

        # === 빈 공간 판정 파라미터 ===
        self.anchor_xy     = np.array([0.0, 0.0], dtype=float)  # 앵커 좌표(기본 0,0)
        self.empty_gap_m   = 3.0                                # 인접 점 사이 3m 이상 → 빈 공간

        # === 내부 상태 ===
        # 트랙: [{'xy': np.array([x,y]), 'count':int, 'miss':int}, ...]
        self.tracks = []

        # 안정 좌표(매 프레임 재구성): np.ndarray (M,2)
        self.stable_pts = np.empty((0, 2), dtype=float)

        # PCA 축 (안정 좌표 존재할 때 갱신)
        self.axis_ready = False
        self.axis_u = np.array([1.0, 0.0], dtype=float)   # 단위 방향(가로)
        self.axis_o = np.array([0.0, 0.0], dtype=float)   # 기준점(평균)
        self.axis_ema_alpha = 0.35

        # 정렬/빈공간 결과
        self.sorted_pts = np.empty((0, 2), dtype=float)
        self.empty_gap_found = False
        self.empty_gap_pair = None   # (pL, pR)

        # 이후 단계용(주차 좌표 생성 훅)
        self.stop_xy = None
        self.parking_xy = None

        # === ROS Pub/Sub ===
        self.parking_stop_done_pub = rospy.Publisher("/parking_stop_done", Int16, queue_size=1)
        self.parking_path_pub      = rospy.Publisher("/parking_path",      PathReference, queue_size=1)
        self.parking_done_pub      = rospy.Publisher("/parking_done",      Int16, queue_size=1)
        self.parking_end_pub       = rospy.Publisher("/parking_end",       Int16, queue_size=1)

        rospy.Subscriber("/obstacle_info", ObjectInfo, self.obstacle_info_CB, queue_size=1)
        rospy.Subscriber("/current_pose",  Pose,       self.curr_pose_CB,     queue_size=1)

        self.curr_pose = None

    # -------------------- 콜백 --------------------

    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg

    def obstacle_info_CB(self, msg: ObjectInfo):
        if self.curr_pose is None:
            return

        # 1) 유효 관측 추출 (차량 좌표 → ROI 필터)
        N = int(msg.objectCounts)
        if N <= 0:
            return

        cx = np.asarray(msg.centerX[:N], dtype=float)
        cy = np.asarray(msg.centerY[:N], dtype=float)

        valid = np.isfinite(cx) & np.isfinite(cy)
        if not valid.any():
            return
        cx, cy = cx[valid], cy[valid]

        d2_local = cx * cx + cy * cy
        roi_mask = d2_local <= (self.ROI_radius ** 2)
        if not roi_mask.any():
            return
        cx, cy = cx[roi_mask], cy[roi_mask]

        # 차량좌표 → 맵좌표
        obs_pts = self.local_to_global(cx, cy)

        # 2) 트랙 매칭 & count 누적
        self.update_cone_state(obs_pts)

        # 3) 안정 좌표(= count >= stable_k) 배열 구성
        self.update_visit_state()

        if self.stable_pts.shape[0] < 2:
            # 빈공간 판정까지는 최소 2점 필요
            self.empty_gap_found = False
            self.empty_gap_pair = None
            return

        # 4) 축 추정 후, 앵커 좌표(self.anchor_xy)로부터 가까운 순 정렬
        self.update_cone_array(self.stable_pts)

        # 5) 인접 원소 거리로 빈 공간 판정 (3m 이상)
        gap_idx = self.find_parking_slot(self.sorted_pts, self.empty_gap_m)
        if gap_idx is not None:
            pL = self.sorted_pts[gap_idx]
            pR = self.sorted_pts[gap_idx + 1]
            self.empty_gap_found = True
            self.empty_gap_pair = (pL.copy(), pR.copy())
            self.parking_coords = pL
        else:
            self.empty_gap_found = False
            self.empty_gap_pair = None

    # -------------------- 좌표계/트래킹 --------------------

    def local_to_global(self, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
        px = float(self.curr_pose.position.x)
        py = float(self.curr_pose.position.y)
        q = self.curr_pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        c, s = math.cos(yaw), math.sin(yaw)

        x_g = px + (cx * c - cy * s)
        y_g = py + (cx * s + cy * c)
        return np.column_stack((x_g, y_g))

    def update_cone_state(self, obs_pts: np.ndarray):
        used = np.zeros(len(self.tracks), dtype=bool)

        for point in obs_pts:
            j = self.find_nearest_coords(point)
            if j is not None and np.linalg.norm(self.tracks[j]['xy'] - point) <= self.match_gate:
                # 매칭 성공 → 좌표 EMA + count++
                self.tracks[j]['xy'] = 0.6 * self.tracks[j]['xy'] + 0.4 * point
                self.tracks[j]['count'] += 1
                self.tracks[j]['miss'] = 0
                used[j] = True
            else:
                # 신규 트랙
                self.tracks.append({'xy': point.copy(), 'count': 1, 'miss': 0})
                used = np.append(used, True)  # 방금 추가된 트랙은 사용 처리

        # 미사용 트랙 miss++
        for k in range(len(self.tracks)):
            if not used[k]:
                self.tracks[k]['miss'] += 1

        # 오래된 트랙 제거
        self.tracks = [t for t in self.tracks if t['miss'] <= self.decay_miss]

    def find_nearest_coords(self, p: np.ndarray):
        if not self.tracks:
            return None
        dists = [np.linalg.norm(t['xy'] - p) for t in self.tracks]
        j = int(np.argmin(dists))
        return j

    def update_visit_state(self):
        pts = [t['xy'] for t in self.tracks if t['count'] >= self.stable_k]
        if pts:
            self.stable_pts = np.vstack(pts).astype(float)
        else:
            self.stable_pts = np.empty((0, 2), dtype=float)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    # -------------------- 4) 축 추정 --------------------

    def update_cone_array(self, pts: np.ndarray):
        if pts.shape[0] < 2:
            return

        mu = pts.mean(axis=0)
        X = pts - mu
        C = X.T @ X
        eigvals, eigvecs = np.linalg.eigh(C)
        u = eigvecs[:, np.argmax(eigvals)]
        u = u / (np.linalg.norm(u) + 1e-12)

        # 방향 부호 일관성
        if np.dot(self.axis_u, u) < 0:
            u = -u

        a = float(self.axis_ema_alpha)
        if self.axis_ready:
            self.axis_u = self._normalize((1 - a) * self.axis_u + a * u)
            self.axis_o = (1 - a) * self.axis_o + a * mu
        else:
            self.axis_u = u
            self.axis_o = mu
            self.axis_ready = True

        # ✅ 앵커 기준 가까운 순 정렬 (pts 사용 + 제곱거리)
        diff = pts - self.anchor_xy            # (N,2)
        d2 = np.einsum('ij,ij->i', diff, diff) # squared distances
        order = np.argsort(d2)
        self.sorted_pts = pts[order]

    # -------------------- 5) 빈 공간 판정 --------------------

    def find_parking_slot(self, sorted_pts: np.ndarray, gap_thres_m: float):
        if sorted_pts.shape[0] < 2:
            return None
        seg = np.linalg.norm(sorted_pts[1:] - sorted_pts[:-1], axis=1)
        idx = np.where(seg >= float(gap_thres_m))[0]
        if idx.size == 0:
            return None
        return int(idx[0])
