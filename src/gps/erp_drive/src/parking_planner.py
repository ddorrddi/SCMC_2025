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
        self.match_gate = 0.30     # 트랙 매칭 허용 거리 (m)
        self.stable_k   = 10       # count 임계값 (>= 이면 안정 좌표 편입)
        self.decay_miss = 10       # 미관측 허용 프레임(지나면 트랙 삭제)

        # === 빈 공간 판정 파라미터 ===
        self.benchmark_coords     = np.array([0.0, 0.0], dtype=float)  # 앵커 좌표(기본 0,0)
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
        self.parking_slot_found = False

        # 주차 좌표 생성
        self.parking_stop_coords = None  # 정지 좌표 (x,y)
        self.reference_coords = None
        self.parking_coords = None
        self.parking_radius = 1.5  # 주차 좌표 생성 반경 (m) -> 실험적으로 찾아야함
        self.parking_theta = 30.0 * math.pi / 180.0  # 주차 좌표 생성 각도 (rad) -> 실험적으로 찾아야함

        # === 주차 경로 생성 파라미터 ===
        self.resolution = 0.3      # 경로 해상도 (m)
        self.flip_arccos = False   # arccos 함수 상하 반전 여부
        self.parking_in_progress = False
        self.parking_path_is_pub = False
        self.parking_path = PathReference()

        # === 주차 상태 플래그 ===
        self.parking_stop_done = False   # 정지 완료 플래그
        self.parking_done = False        # 주차 완료 플래그
        self.parking_end = False         # 주차 종료 플래그
        self.parking_done_once = False   # 주차 완료 1회 체크용 플래그
        self.return_phase = 0            # g_path 복귀 단계 플래그 (0: 대기, 1: 경로 발행, 2: 복귀 완료 대기)
        self.Parking_Stop_Radius = 2.0   # 정지 반경 (m)
        self.Parking_Success_Radius = 0.5  # 주차 성공 반경 (m)

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
        self.parking_stop_confirm()

        if self.parking_stop_done and not self.parking_path_is_pub:
            self.make_parking_path()

        if not self.parking_end and (self.parking_path_is_pub or self.return_phase > 0):
            self.return_to_gpath()

    def obstacle_info_CB(self, msg: ObjectInfo):
        if self.curr_pose is None:
            return

        if not self.parking_slot_found:
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
                self.parking_slot_found = False
                return

            # 4) 축 추정 후, 앵커 좌표(self.benchmark_coords)로부터 가까운 순 정렬
            self.update_cone_array(self.stable_pts)

            # 5) 빈 공간 판정 후 좌표 생성 (우측으로만 생성)
            gap_idx = self.find_parking_slot(self.sorted_pts, self.empty_gap_m)
            if gap_idx is not None:
                ref  = self.sorted_pts[gap_idx]
                stop = self.sorted_pts[gap_idx + 1]

                # 레일 축 단위 벡터(접선) & 우측 법선(시계방향 90° 회전)
                t_hat   = self.axis_u / (np.linalg.norm(self.axis_u) + 1e-12)
                n_right = np.array([t_hat[1], -t_hat[0]], dtype=float)   # right normal (CW 90°)

                # 축 기준 상대각으로 방향 구성 (θ=0: 축 방향, θ>0: 우측으로 기울임)
                v_dir = math.cos(self.parking_theta) * t_hat + math.sin(self.parking_theta) * n_right
                v_dir /= (np.linalg.norm(v_dir) + 1e-12)

                self.reference_coords    = ref
                self.parking_stop_coords = stop            # 그대로 한쪽 콘을 정지 기준으로 사용
                self.parking_coords      = ref + self.parking_radius * v_dir

                self.parking_slot_found  = True
                self.parking_in_progress = True
            else:
                self.parking_slot_found = False


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

        # 방향 부호 일관성 유지
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

        # --- 축 방향 스칼라로 정렬 ---
        proj = (pts - self.axis_o) @ self.axis_u  # (N,)
        order = np.argsort(proj)
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
    
    # -------------------- 주차 경로 생성 --------------------
    def make_parking_path(self):
        if self.parking_coords is None or self.curr_pose is None or self.parking_slot_found is False:
            return

        start_xy = np.array([self.curr_pose.position.x, self.curr_pose.position.y], dtype=float)
        goal_xy  = self.parking_coords.astype(float)

        raw = self.make_raw_path(start_xy, goal_xy, n=800, flip=self.flip_arccos)
        pts = self.spacing_parking_path(raw, self.resolution)

        self.parking_path.header.frame_id = self.frame_id
        self.parking_path.header.stamp = rospy.Time.now()
        self.parking_path.path.clear()

        append = self.parking_path.path.append
        for x, y in pts:
            elem = PathReferenceElement()
            p = elem.pose
            p.position.x = float(x); p.position.y = float(y)
            p.orientation.w = 1.0
            append(elem)

        if self.parking_in_progress and not self.parking_path_is_pub:
            self.parking_path_pub.publish(self.parking_path)
            self.parking_path_is_pub = True

    def make_raw_path(self, start_xy, goal_xy, n=800, flip=False):
        start_xy = np.asarray(start_xy, dtype=float)
        goal_xy  = np.asarray(goal_xy,  dtype=float)

        A = np.array([-1.0, np.pi], dtype=float)
        B = np.array([ 1.0, 0.0 ], dtype=float)

        x = np.linspace(-1.0, 1.0, n)
        y = np.arccos(x)
        if flip:
            y = -y
        P = np.column_stack([x, y])

        v_ab = B - A
        v    = goal_xy - start_xy
        s = np.linalg.norm(v) / np.linalg.norm(v_ab)

        ang_ab = np.arctan2(v_ab[1], v_ab[0])
        ang_v  = np.arctan2(v[1],    v[0])
        th = ang_v - ang_ab
        c, si = np.cos(th), np.sin(th)
        R = np.array([[c, -si],[si, c]], dtype=float)
        t = start_xy - s * (R @ A)

        return (s * (R @ P.T)).T + t
    
    def spacing_parking_path(self, pts, spacing):
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return pts.copy()
        seg   = pts[1:] - pts[:-1]
        segln = np.linalg.norm(seg, axis=1)
        s     = np.concatenate([[0.0], np.cumsum(segln)])
        total = s[-1]
        if total <= 1e-9:
            return pts[[0]].copy()
        M = int(np.floor(total / spacing)) + 1
        target = np.linspace(0.0, total, M)
        out = np.empty((M, 2), dtype=float)
        j = 0
        for i, ts in enumerate(target):
            while j < len(s) - 2 and s[j+1] < ts:
                j += 1
            if s[j+1] == s[j]:
                out[i] = pts[j]
            else:
                a = (ts - s[j]) / (s[j+1] - s[j])
                out[i] = (1 - a) * pts[j] + a * pts[j+1]
        return out
    
    # -------------------- 주차 후 g_path로 복귀 --------------------

    def return_to_gpath(self):
        if self.return_phase == 0:
            self.parking_done_confirm()
            if self.parking_done:
                self.return_phase = 1
            return

        if self.return_phase == 1:
            path_list = self.parking_path.path
            if self.parking_done and len(path_list) >= 2:
                path_list.reverse()
                self.parking_path.header.stamp = rospy.Time.now()
                self.parking_path_pub.publish(self.parking_path)
                self.return_phase = 2
            return

        if self.return_phase == 2:
            self.return_done_confirm()
            return
        
    # -------------------- 상태 pub --------------------

    def parking_stop_confirm(self):
        if self.parking_stop_coords is None or self.curr_pose is None:
            self.parking_stop_done = False
            return
        
        parking_stop_x, parking_stop_y = self.parking_stop_coords

        dist_to_target = np.hypot(self.curr_pose.position.x - parking_stop_x,\
                                   self.curr_pose.position.y - parking_stop_y)
        
        if not self.parking_done_once:
            if dist_to_target < self.Parking_Stop_Radius and self.parking_in_progress:
                self.parking_stop_done = True
                self.parking_done_once = True

        self.parking_stop_done_pub.publish(1 if self.parking_stop_done else 0)

    def parking_done_confirm(self):
        path_list = self.parking_path.path

        if self.curr_pose is None or not path_list:
            return
        
        pose = getattr(path_list[-1].pose, "pose", path_list[-1].pose)
        end_x, end_y = float(pose.position.x), float(pose.position.y)
        px, py = float(self.curr_pose.position.x), float(self.curr_pose.position.y)

        self.parking_done = (math.hypot(px - end_x, py - end_y) <= self.Parking_Success_Radius)
        self.parking_done_pub.publish(1 if self.parking_done else 0)

    def return_done_confirm(self):
        path_list = self.parking_path.path
        if self.curr_pose is None or not path_list:
            return
        pose = getattr(path_list[-1].pose, "pose", path_list[-1].pose)
        end_x, end_y = float(pose.position.x), float(pose.position.y)
        px, py = float(self.curr_pose.position.x), float(self.curr_pose.position.y)

        if math.hypot(px - end_x, py - end_y) <= self.Parking_Success_Radius:
            self.parking_end = True
            self.parking_end_pub.publish(1 if self.parking_end else 0)
            self.parking_in_progress = False
            self.return_phase = 3