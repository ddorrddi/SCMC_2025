#!/usr/bin/env python3
import math
import numpy as np
import rospy

from geometry_msgs.msg import Pose, PoseStamped, Point
from lidar_object_detection.msg import ObjectInfo
from std_msgs.msg import Int16, Bool
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

from erp_drive.msg import PathReference, PathReferenceElement

class ParkingPlanner:
    def __init__(self, rviz_offset_x: float, rviz_offset_y: float):
        # === RViz 보조(그대로) ===
        self.rviz_offset_x = float(rviz_offset_x)
        self.rviz_offset_y = float(rviz_offset_y)

        # === 트래킹/안정화 파라미터 ===
        self.match_gate = 0.30     # 트랙 매칭 허용 거리 (m)

        # === 빈 공간 판정 파라미터 ===
        self.benchmark_coords = np.array([935545.1301428889, 1915881.503757789], dtype=float)  # 앵커 좌표

        # self.benchmark_coords = np.array([935562.8970950138, 1915800.5427482247], dtype=float)  # 앵커 좌표



        # === 장애물 인지 및 트래킹 ===
        self.tracks = []
        self.stable_pts = np.empty((0, 2), dtype=float)
        self.sorted_pts = np.empty((0, 2), dtype=float)

        # PCA 축
        self.axis_ready = False
        self.axis_u = np.array([1.0, 0.0], dtype=float)   # 단위 방향(가로)
        self.axis_o = np.array([0.0, 0.0], dtype=float)   # 기준점(평균)
        self.axis_ema_alpha = 0.35

        # 주차 좌표 생성
        self.parking_stop_coords = None  # 정지 좌표 (x,y)
        self.reference_coords = None
        self.parking_start_coords = None  # 진입 좌표 (x,y)
        self.parking_coords = None

        # === 주차 경로 생성 파라미터 ===
        self.resolution = 0.1      # 경로 해상도 (m)
        self.path_delay_sec = 14.0 # 경로 발행 지연 시간 (s) -> 주차 정지 후 약간의 시간이 지난 후 경로 발행
        self.path_timer = None
        self.flip_arccos = False   # arccos 함수 상하 반전 여부
        self.parking_path = PathReference()
        self.return_path = PathReference()
        self.rviz_parking_path = Path()
        self.rviz_return_path = Path()

        # === 주차 상태 플래그 ===
        self.parking_slot_found = False
        self.parking_in_progress = False
        self.parking_stop_done = False     # 정지 완료 플래그
        self.parking_path_is_pub = False   # 주차 경로 발행 플래그
        self.parking_done = False          # 주차 완료 플래그
        self.parking_done_once = False     # 주차 완료 1회 체크용 플래그
        self.parking_end = False           # 주차 종료 플래그
        self.return_phase = 0              # g_path 복귀 단계 플래그 (0: 대기, 1: 경로 발행, 2: 복귀 완료 대기)

        self.curr_pose = None
        self.parking_path_set = False
        self.curr_idx = 0


        # === 튜닝 파라미터 ===
        self.ROI_radius = 3.0      # 차량 기준 ROI (m)
        self.empty_gap = 4.0       # 인접 라바콘 사이 거리 4m 이상 → 빈 공간
        self.stable_k   = 1        # count 임계값 (>= 이면 안정 좌표 편입)   ->   근데 이건 나중에 실제 라이다 돌리면서 기준 잡아야함
        self.decay_miss = 10       # 미관측 허용 프레임(지나면 트랙 삭제)   ->  이거도 나중에 실제 라이다 돌리면서 기준 잡아야함
        self.parking_radius = 3.0  # 주차 좌표 생성 반지름 (m)
        self.parking_theta = 161.0 * math.pi / 180.0 # 주차 좌표 생성 각도 (rad)
        self.Parking_Stop_Radius = 2.2     # 정지 반경 (m)
        self.Parking_Success_Radius = 0.5  # 주차 성공 반경 (m)
        self.Return_Success_Radius = 2.0

        # === ROS Pub/Sub ===
        self.parking_stop_done_pub = rospy.Publisher("/parking_stop_done", Int16,         queue_size=1)
        self.parking_done_pub      = rospy.Publisher("/parking_done",      Int16,         queue_size=1)
        self.parking_end_pub       = rospy.Publisher("/parking_end",       Int16,         queue_size=1)
        self.parking_path_pub      = rospy.Publisher("/parking_path",      PathReference, queue_size=1)
        self.return_path_pub       = rospy.Publisher("/return_path",       PathReference, queue_size=1)
        self.rviz_parking_path_start_point_pub = rospy.Publisher("/rviz_parking_start_point", PoseStamped,          queue_size=1)
        self.rviz_parking_path_pub = rospy.Publisher("/rviz_parking_path", Path,          queue_size=1)
        self.rviz_return_path_pub  = rospy.Publisher("/rviz_return_path",  Path,          queue_size=1)
        self.sorted_pts_marker_pub = rospy.Publisher("/sorted_pts_marker", Marker,        queue_size=1)

        rospy.Subscriber("/object_info_labacon", ObjectInfo,  self.obstacle_info_CB, queue_size=1)
        rospy.Subscriber("/current_pose",  Pose,        self.curr_pose_CB,     queue_size=1)
        rospy.Subscriber("/curr_idx", Int16, self.curr_idx_CB, queue_size=1)

    # -------------------- 콜백 --------------------

    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg
        
        if self.parking_slot_found and not self.parking_done:
            self.parking_stop_confirm()

        if not self.parking_end and self.parking_path_is_pub:
            self.return_to_gpath()

        self.publish_sorted_pts_marker()

    def curr_idx_CB(self, msg: Int16):

        self.curr_idx = msg.data

    def obstacle_info_CB(self, msg: ObjectInfo):
        if self.curr_pose is None:
            return

        if self.curr_idx < 3070: # 3040->3070 근데 3070이면 좀 늦을거 같음
            return

        if not self.parking_slot_found:
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

            # 좌표계 변환
            obs_pts = self.local_to_global(cx, cy)

            # 2) 매칭 및 count 누적
            self.update_labacon_state(obs_pts)

            # 3) 안정 좌표 리스트 삽입
            self.update_stable_state()

            if self.stable_pts.shape[0] < 2:
                self.parking_slot_found = False
                return

            # 4) 축 추정 및 정렬
            self.update_labacon_array(self.stable_pts)

            # 5) 빈 공간 판정 후 좌표 생성
            self.generate_parking_coords()

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

    def update_labacon_state(self, obs_pts: np.ndarray):
        used = np.zeros(len(self.tracks), dtype=bool)

        for point in obs_pts:
            j = self.find_nearest_coords(point)
            if j is not None and not used[j] and np.linalg.norm(self.tracks[j]['xy'] - point) <= self.match_gate:    # 매칭 성공 → 좌표 EMA + count++
                self.tracks[j]['xy'] = 0.6 * self.tracks[j]['xy'] + 0.4 * point
                self.tracks[j]['count'] += 1
                self.tracks[j]['miss'] = 0
                used[j] = True

            else:   # 신규 트랙
                self.tracks.append({'xy': point.copy(), 'count': 1, 'miss': 0})
                used = np.append(used, True)

        # 미사용 트랙 miss++
        for k in range(len(self.tracks)):
            if not used[k]:
                self.tracks[k]['miss'] += 1

        # 오래된 트랙 제거
        self.tracks = [t for t in self.tracks if t['miss'] <= self.decay_miss]

    def find_nearest_coords(self, p: np.ndarray):
        if not self.tracks:
            return
        dists = [np.linalg.norm(t['xy'] - p) for t in self.tracks]
        j = int(np.argmin(dists))
        return j

    def update_stable_state(self):
        pts = [t['xy'] for t in self.tracks if t['count'] >= self.stable_k]
        if pts:
            self.stable_pts = np.vstack(pts).astype(float)
        else:
            self.stable_pts = np.empty((0, 2), dtype=float)

    # -------------------- 4) 축 추정 --------------------

    def normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v
    
    def update_labacon_array(self, pts: np.ndarray):
        if pts.shape[0] == 0:
            self.sorted_pts = np.empty((0, 2), dtype=float)
            return

        # PCA로 축 추정 및 EMA로 안정화
        if pts.shape[0] >= 2:
            mu = pts.mean(axis=0)
            X  = pts - mu
            C  = X.T @ X
            eigvals, eigvecs = np.linalg.eigh(C)
            u = eigvecs[:, np.argmax(eigvals)]
            u = u / (np.linalg.norm(u) + 1e-12)

            if np.dot(self.axis_u, u) < 0:
                u = -u

            a = float(self.axis_ema_alpha)
            if self.axis_ready:
                self.axis_u = self.normalize((1 - a) * self.axis_u + a * u)
                self.axis_o = (1 - a) * self.axis_o + a * mu
            else:
                self.axis_u = u
                self.axis_o = mu
                self.axis_ready = True

        if pts.shape[0] >= 2 and self.axis_ready:
            s  = (pts - self.axis_o) @ self.axis_u
            s0 = (np.asarray(self.benchmark_coords) - self.axis_o) @ self.axis_u
            order = np.argsort(np.abs(s - s0))
        else:
            bc = np.asarray(self.benchmark_coords, dtype=float)
            d2 = np.sum((pts - bc[None, :])**2, axis=1)
            order = np.argsort(d2)

        self.sorted_pts = pts[order]

    # -------------------- 5) 빈 공간 판정 --------------------

    def find_parking_slot(self, sorted_pts):
        if sorted_pts.shape[0] < 2:
            return
        
        seg = np.linalg.norm(sorted_pts[1:] - sorted_pts[:-1], axis=1)
        cand = np.where(seg >= float(self.empty_gap))[0]
        if cand.size == 0:
            return

        k = cand[np.argmax(seg[cand])]
        return int(k)

    # -------------------- 6) 좌표 생성 --------------------

    def generate_parking_coords(self):
        gap_idx = self.find_parking_slot(self.sorted_pts)
        if gap_idx is not None:
            stop = self.sorted_pts[gap_idx + 1]

            # 차량 진행방향(헤딩)과 차량 우측 벡터
            q = self.curr_pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            fwd   = np.array([math.cos(yaw),  math.sin(yaw)], dtype=float)
            right = np.array([ math.sin(yaw), -math.cos(yaw)], dtype=float)

            # 진행방향을 기준으로 오른쪽으로 θ만큼 기울인 단위 벡터
            v_dir = math.cos(self.parking_theta) * fwd + math.sin(self.parking_theta) * right
            v_dir /= (np.linalg.norm(v_dir) + 1e-12)

            self.parking_stop_coords = stop
            self.parking_coords      = stop + self.parking_radius * v_dir

            self.parking_slot_found  = True
            self.parking_in_progress = True
        else:
            self.parking_slot_found = False

    # -------------------- 주차 경로 생성 --------------------

    def make_parking_path(self):
        if self.parking_coords is None or self.curr_pose is None or self.parking_slot_found is False:
            return

        raw = self.make_raw_path(n=800, flip=self.flip_arccos)
        pts = self.spacing_parking_path(raw, self.resolution)

        self.parking_path.path.clear()
        self.rviz_parking_path.poses.clear()

        for x, y in pts:
            parking_path_element = PathReferenceElement()
            parking_path_element.pose.position.x = x
            parking_path_element.pose.position.y = y
            parking_path_element.pose.position.z = 0.
            parking_path_element.pose.orientation.x = 0.
            parking_path_element.pose.orientation.y = 0.
            parking_path_element.pose.orientation.z = 0.
            parking_path_element.pose.orientation.w = 1.
            parking_path_element.mode = 11
            self.parking_path.path.append(parking_path_element)

            rviz_parking_pose = PoseStamped()
            rviz_parking_pose.header.frame_id = "map"
            rviz_parking_pose.header.stamp = rospy.Time.now()
            rviz_parking_pose.pose.position.x = parking_path_element.pose.position.x - self.rviz_offset_x
            rviz_parking_pose.pose.position.y = parking_path_element.pose.position.y - self.rviz_offset_y
            rviz_parking_pose.pose.position.z = 0.
            rviz_parking_pose.pose.orientation.x = 0.
            rviz_parking_pose.pose.orientation.y = 0.
            rviz_parking_pose.pose.orientation.z = 0.
            rviz_parking_pose.pose.orientation.w = 1.
            self.rviz_parking_path.poses.append(rviz_parking_pose)

        self.rviz_parking_path.header.stamp = rospy.Time.now()
        self.rviz_parking_path.header.frame_id = "map"

        if self.parking_in_progress and not self.parking_path_is_pub:
            self.parking_path_pub.publish(self.parking_path)
            self.rviz_parking_path_pub.publish(self.rviz_parking_path)
            self.rviz_parking_path_start_point_pub.publish(self.rviz_parking_path.poses[0])
            self.parking_path_is_pub = True
            rospy.loginfo(f"Parking path published!!!!!!")

    # -------------------- rev 경로 생성 --------------------

    def generate_return_path(self):
        rospy.loginfo(f"Start generating return path!!!!!!")
        path = self.parking_path.path
        if len(path) < 2:
            return
        
        self.return_path.path.clear()
        self.rviz_return_path.poses.clear()

        pts = [(e.pose.position.x, e.pose.position.y) for e in path]
        keep_count = max(0, int(len(pts) - 5))
        if keep_count == 0:
            return
        pts = pts[:keep_count][::-1]

        for x, y in pts:
            return_path_element = PathReferenceElement()
            return_path_element.pose.position.x = x
            return_path_element.pose.position.y = y
            return_path_element.pose.position.z = 0.
            return_path_element.pose.orientation.x = 0.
            return_path_element.pose.orientation.y = 0.
            return_path_element.pose.orientation.z = 0.
            return_path_element.pose.orientation.w = 1.
            return_path_element.mode = 11
            self.return_path.path.append(return_path_element)

            rviz_return_pose = PoseStamped()
            rviz_return_pose.header.frame_id = "map"
            rviz_return_pose.header.stamp = rospy.Time.now()
            rviz_return_pose.pose.position.x = return_path_element.pose.position.x - self.rviz_offset_x
            rviz_return_pose.pose.position.y = return_path_element.pose.position.y - self.rviz_offset_y
            rviz_return_pose.pose.position.z = 0.
            rviz_return_pose.pose.orientation.x = 0.
            rviz_return_pose.pose.orientation.y = 0.
            rviz_return_pose.pose.orientation.z = 0.
            rviz_return_pose.pose.orientation.w = 1.
            self.rviz_return_path.poses.append(rviz_return_pose)

        self.rviz_return_path.header.stamp = rospy.Time.now()
        self.rviz_return_path.header.frame_id = "map"

        self.return_path_pub.publish(self.return_path)
        self.rviz_return_path_pub.publish(self.rviz_return_path)
        rospy.loginfo(f"Return path published!!!!!!")
        
    # -------------------- 딜레이 경로 생성 --------------------

    def delay_path_maker(self, event=None):
        if self.parking_in_progress and self.parking_stop_done and not self.parking_path_is_pub:
            self.make_parking_path()
        if self.parking_in_progress and self.parking_done and not self.parking_end:
            self.generate_return_path()
        
        self.path_timer = None

    # -------------------- raw path 생성 --------------------

    def make_raw_path(self, n=800, flip=False, ext_length=4.0, ext_points=None):
        import numpy as np

        # 안전한 시작/목표 추출 (네가 올려준 코드 가정)
        start_x = (float(self.curr_pose.position.x)*4.0 + float(self.parking_start_coords[0]))/5.0
        start_y = (float(self.curr_pose.position.y)*4.0 + float(self.parking_start_coords[1]))/5.0
        goal_x  = float(self.parking_coords[0])
        goal_y  = float(self.parking_coords[1])

        start_xy = np.array([start_x, start_y], dtype=float)
        goal_xy  = np.array([goal_x,  goal_y ], dtype=float)

        A = np.array([-1.0, np.pi], dtype=float)
        B = np.array([ 1.0, 0.0 ], dtype=float)

        x = np.linspace(-1.0, 1.0, int(n))
        y = np.arccos(x)
        if flip:
            y = -y
        P = np.column_stack([x, y])

        v_ab = B - A
        v    = goal_xy - start_xy
        # 안전: 0으로 나누는 것 방지
        norm_vab = np.linalg.norm(v_ab)
        if norm_vab < 1e-12:
            # fallback: 직선 보간
            tlin = np.linspace(0.0, 1.0, int(n))
            line = np.column_stack([start_xy[0] + (goal_xy[0]-start_xy[0]) * tlin,
                                    start_xy[1] + (goal_xy[1]-start_xy[1]) * tlin])
            return line

        s = np.linalg.norm(v) / norm_vab

        ang_ab = np.arctan2(v_ab[1], v_ab[0])
        ang_v  = np.arctan2(v[1],    v[0])
        th = ang_v - ang_ab
        c, si = np.cos(th), np.sin(th)
        R = np.array([[c, -si],[si, c]], dtype=float)
        t = start_xy - s * (R @ A)

        # 기본 변환된 경로 (N x 2)
        pts = (s * (R @ P.T)).T + t
        pts = np.asarray(pts, dtype=float)

        # 만약 포인트가 2개 미만이면 연장 불가 -> 그대로 반환
        if pts.shape[0] < 2:
            return pts

        # 직진 연장 파라미터
        if ext_points is None:
            ext_points = max(5, int(max(1, n * 0.05)))  # 기본: 전체의 5% 또는 최소 5개
        ext_points = int(ext_points)
        ext_length = float(ext_length)

        # 마지막 방향 벡터: 마지막 두 점 이용 (안정성: 너무 작으면 ang_v 사용)
        last_vec = pts[-1] - pts[-2]
        last_norm = np.linalg.norm(last_vec)
        if last_norm < 1e-9:
            # 거의 동일한 점이면 ang_v 사용
            dir_vec = np.array([np.cos(ang_v), np.sin(ang_v)], dtype=float)
        else:
            dir_vec = last_vec / last_norm

        # ext 구간 t (0 excluded because last point already 포함)
        # we generate ext_points samples along (0, ext_length], excluding 0 to avoid duplicate
        ext_t = np.linspace(ext_length / ext_points, ext_length, ext_points)
        ext_pts = pts[-1] + np.outer(ext_t, dir_vec)  # shape (ext_points, 2)

        # 합치기
        combined = np.vstack([pts, ext_pts])

        return combined

    
    # -------------------- 등간격 보간 --------------------

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
            if self.parking_done and len(self.parking_path.path) >= 2:
                if self.path_timer is None:
                    rospy.loginfo(f"delay path maker!!!!!!")
                    self.path_timer = rospy.Timer(rospy.Duration(4.0), self.delay_path_maker, oneshot=True)  # 타이머 취소 후 바로 경로 생성 예약
                self.return_phase = 2
            return

        if self.return_phase == 2:
            self.return_done_confirm()
            return
        
    # -------------------- 상태 pub --------------------

    def parking_stop_confirm(self):
        if self.parking_stop_coords is None or self.curr_pose is None or not self.parking_in_progress:
            self.parking_stop_done = False
            return
        
        parking_stop_x, parking_stop_y = self.parking_stop_coords

        dist_to_target = np.hypot(self.curr_pose.position.x - parking_stop_x, self.curr_pose.position.y - parking_stop_y)
        
        if not self.parking_done_once:
            if dist_to_target < self.Parking_Stop_Radius and self.parking_in_progress:
                self.parking_start_coords = np.array([self.curr_pose.position.x, self.curr_pose.position.y], dtype=float)
                self.parking_stop_done = True
                self.parking_done_once = True
                rospy.loginfo(f"Parking Stop Done!!!!!!!!!")

                if self.path_timer is None:
                    rospy.loginfo(f"delay path maker start!!!!!!")
                    self.path_timer = rospy.Timer(rospy.Duration(self.path_delay_sec), self.delay_path_maker, oneshot=True)     #1.5초 뒤 경로 생성 예약

        self.parking_stop_done_pub.publish(1 if self.parking_stop_done else 0)

    def parking_done_confirm(self):
        path_list = self.parking_path.path
        if self.curr_pose is None or not path_list:
            return
        
        end_x, end_y = float(self.parking_coords[0]), float(self.parking_coords[1])
        px, py = float(self.curr_pose.position.x), float(self.curr_pose.position.y)
        
        self.parking_done = (math.hypot(px - end_x, py - end_y) <= self.Parking_Success_Radius)
        self.parking_done_pub.publish(1 if self.parking_done else 0)
        if self.parking_done:
            rospy.loginfo(f"Parking Done!!!!!!!!!")

    def return_done_confirm(self):
        path_list = self.return_path.path
        if self.curr_pose is None or not path_list:
            return

        last_element = path_list[-1]
        end_x, end_y = float(last_element.pose.position.x), float(last_element.pose.position.y)
        px, py = float(self.curr_pose.position.x), float(self.curr_pose.position.y)

        if math.hypot(px - end_x, py - end_y) <= self.Return_Success_Radius:
            self.parking_end = True
            self.parking_end_pub.publish(1 if self.parking_end else 0)
            if self.parking_end:
                rospy.loginfo(f"Return to g_path Done!!!!!!!!!")
            self.parking_in_progress = False
            self.return_phase = 3

    def publish_sorted_pts_marker(self):
        if self.sorted_pts is None or self.sorted_pts.shape[0] == 0:
            return

        now = rospy.Time.now()
        frame_id = "map"

        def P(x, y, z=0.0):
            return Point(x=float(x) - self.rviz_offset_x, y=float(y) - self.rviz_offset_y, z=float(z))

        pts = [P(x, y) for x, y in self.sorted_pts]

        # --- 정렬 점들 (흰색) ---
        m_pts = Marker()
        m_pts.header.frame_id = frame_id
        m_pts.header.stamp = now
        m_pts.ns = "sorted_pts"
        m_pts.id = 0
        m_pts.type = Marker.SPHERE_LIST
        m_pts.action = Marker.ADD
        m_pts.scale.x = 0.22
        m_pts.scale.y = 0.22
        m_pts.scale.z = 0.22
        m_pts.color.r = 1.00
        m_pts.color.g = 1.00
        m_pts.color.b = 1.00
        m_pts.color.a = 0.95
        m_pts.points = pts

        # --- 정렬 라인 (회색) ---
        m_line = Marker()
        m_line.header.frame_id = frame_id
        m_line.header.stamp = now
        m_line.ns = "sorted_pts"
        m_line.id = 1
        m_line.type = Marker.LINE_STRIP
        m_line.action = Marker.ADD
        m_line.scale.x = 0.05
        m_line.color.r = 0.5
        m_line.color.g = 0.5
        m_line.color.b = 0.5
        m_line.color.a = 0.95
        m_line.points = pts

        self.sorted_pts_marker_pub.publish(m_pts)
        self.sorted_pts_marker_pub.publish(m_line)

        # --- gap, gap+1 강조 (빨강 / 파랑) ---
        gap_idx = self.find_parking_slot(self.sorted_pts) if self.sorted_pts.shape[0] >= 2 else None
        if gap_idx is not None and gap_idx + 1 < self.sorted_pts.shape[0]:
            gx, gy = self.sorted_pts[gap_idx]
            hx, hy = self.sorted_pts[gap_idx + 1]

            gap_marker = Marker()
            gap_marker.header.frame_id = frame_id
            gap_marker.header.stamp = now
            gap_marker.ns = "sorted_pts_gap"
            gap_marker.id = 2
            gap_marker.type = Marker.SPHERE
            gap_marker.action = Marker.ADD
            gap_marker.pose.position = P(gx, gy)
            gap_marker.scale.x = 0.34
            gap_marker.scale.y = 0.34
            gap_marker.scale.z = 0.34
            gap_marker.color.r = 1.00  # 빨강
            gap_marker.color.g = 0.00
            gap_marker.color.b = 0.00
            gap_marker.color.a = 1.00

            gap_next_marker = Marker()
            gap_next_marker.header.frame_id = frame_id
            gap_next_marker.header.stamp = now
            gap_next_marker.ns = "sorted_pts_gap"
            gap_next_marker.id = 3
            gap_next_marker.type = Marker.SPHERE
            gap_next_marker.action = Marker.ADD
            gap_next_marker.pose.position = P(hx, hy)
            gap_next_marker.scale.x = 0.34
            gap_next_marker.scale.y = 0.34
            gap_next_marker.scale.z = 0.34
            gap_next_marker.color.r = 0.00
            gap_next_marker.color.g = 0.00
            gap_next_marker.color.b = 1.00  # 파랑
            gap_next_marker.color.a = 1.00

            self.sorted_pts_marker_pub.publish(gap_marker)
            self.sorted_pts_marker_pub.publish(gap_next_marker)

        # --- 원 그리기 유틸 ---
        def circle_marker(center_xy, radius, mid, r, g, b, ns):
            if center_xy is None or radius is None or radius <= 0:
                return None
            seg = 72
            circle = Marker()
            circle.header.frame_id = frame_id
            circle.header.stamp = now
            circle.ns = ns
            circle.id = mid
            circle.type = Marker.LINE_STRIP
            circle.action = Marker.ADD
            circle.scale.x = 0.06
            circle.color.r = r
            circle.color.g = g
            circle.color.b = b
            circle.color.a = 0.95
            cx, cy = float(center_xy[0]), float(center_xy[1])
            pts_circ = []
            for i in range(seg + 1):
                th = 2.0 * math.pi * i / seg
                x = cx + radius * math.cos(th)
                y = cy + radius * math.sin(th)
                pts_circ.append(P(x, y))
            circle.points = pts_circ
            return circle

        # --- Parking_Stop_Radius (하늘) ---
        stop_circle = circle_marker(self.parking_stop_coords, getattr(self, "Parking_Stop_Radius", 0.0), mid=10, r=0.2, g=0.8, b=1.0, ns="parking_radii")
        if stop_circle is not None:
            self.sorted_pts_marker_pub.publish(stop_circle)

        # --- Parking_Success_Radius (주황) ---
        success_circle = circle_marker(self.parking_coords, getattr(self, "Parking_Success_Radius", 0.0), mid=11, r=1.0, g=0.5, b=0.0, ns="parking_radii")
        
        if success_circle is not None:
            self.sorted_pts_marker_pub.publish(success_circle)
