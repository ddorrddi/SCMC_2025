#!/usr/bin/env python3
import math
import numpy as np
import rospy

from geometry_msgs.msg import Pose, PoseStamped
from lidar_object_detection.msg import ObjectInfo
from std_msgs.msg import Int16, Bool
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Path


from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class ParkingPlanner:
    def __init__(self, rviz_offset_x: float, rviz_offset_y: float):
        # === RViz 보조(그대로) ===
        self.rviz_offset_x = float(rviz_offset_x)
        self.rviz_offset_y = float(rviz_offset_y)

        # === 기본 파라미터 ===
        self.frame_id = "map"
        self.ROI_radius = 1.5  # 차량 기준 ROI (m) -> 실험적으로 찾아야함

        # === 트래킹/안정화 파라미터 ===
        self.match_gate = 0.30     # 트랙 매칭 허용 거리 (m)
        self.stable_k   = 10       # count 임계값 (>= 이면 안정 좌표 편입)
        self.decay_miss = 10       # 미관측 허용 프레임(지나면 트랙 삭제)

        # === 빈 공간 판정 파라미터 ===
        self.benchmark_coords     = np.array([0.0, 0.0], dtype=float)  # 앵커 좌표(기본 0,0)
        self.empty_gap = 3.0       # 인접 점 사이 3m 이상 → 빈 공간

        # === 내부 상태 ===
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
        self.parking_theta = 30.0 * math.pi / 180.0 # 주차 좌표 생성 각도 (rad) -> 실험적으로 찾아야함

        # === 주차 경로 생성 파라미터 ===
        self.resolution = 0.3      # 경로 해상도 (m)
        self.flip_arccos = False   # arccos 함수 상하 반전 여부
        self.parking_in_progress = False
        self.parking_path_is_pub = False
        self.parking_path = Path()
        self.return_path = Path()

        # === 주차 상태 플래그 ===
        self.parking_stop_done = False     # 정지 완료 플래그
        self.parking_done = False          # 주차 완료 플래그
        self.parking_end = False           # 주차 종료 플래그
        self.return_phase = 0              # g_path 복귀 단계 플래그 (0: 대기, 1: 경로 발행, 2: 복귀 완료 대기)
        self.parking_done_once = False     # 주차 완료 1회 체크용 플래그
        self.Parking_Stop_Radius = 2.0     # 정지 반경 (m)
        self.Parking_Success_Radius = 0.5  # 주차 성공 반경 (m)

        # === ROS Pub/Sub ===
        self.parking_stop_done_pub = rospy.Publisher("/parking_stop_done", Int16, queue_size=1)
        self.parking_path_pub      = rospy.Publisher("/parking_path",      Path,  queue_size=1)
        self.return_path_pub       = rospy.Publisher("/return_path",       Path,  queue_size=1)
        self.parking_done_pub      = rospy.Publisher("/parking_done",      Int16, queue_size=1)
        self.parking_end_pub       = rospy.Publisher("/parking_end",       Int16, queue_size=1)

        rospy.Subscriber("/obstacle_info", ObjectInfo,  self.obstacle_info_CB, queue_size=1)
        rospy.Subscriber("/current_pose",  Pose, self.curr_pose_CB,     queue_size=1)

        self.marker_pub = rospy.Publisher("/parking_debug_marker", Marker, queue_size=10)
        rospy.Subscriber('/test_parking', Bool, self.test_CB) # 테스트 서브스크라이버

        self.curr_pose = None

    # -------------------- 테스트 코드 ---------------------
    def test_CB(self, msg: Bool):
        if self.curr_pose is None:
            return
        
        self.sorted_pts = np.array([[935569.271, 1915907.887], [935574.389, 1915916.977]], dtype=float)
        self.generate_parking_coords()

        if self.parking_coords is not None:
            self.make_parking_path()

    # -------------------- 콜백 --------------------

    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg
        self.parking_stop_confirm()

        if self.parking_stop_done and not self.parking_path_is_pub:
            self.make_parking_path()

        if not self.parking_end and self.parking_path_is_pub:
            self.return_to_gpath()

        self.publish_markers()

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
            self.update_stable_state()

            if self.stable_pts.shape[0] < 2:
                # 빈공간 판정까지는 최소 2점 필요
                self.parking_slot_found = False
                return

            # 4) 축 추정 후, 앵커 좌표(self.benchmark_coords)로부터 가까운 순 정렬
            self.update_cone_array(self.stable_pts)

            # 5) 빈 공간 판정 후 좌표 생성 (진행방향의 오른쪽만 사용)
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

    def update_cone_state(self, obs_pts: np.ndarray):
        used = np.zeros(len(self.tracks), dtype=bool)

        for point in obs_pts:
            j = self.find_nearest_coords(point)
            if j is not None \
            and not used[j] \
            and np.linalg.norm(self.tracks[j]['xy'] - point) <= self.match_gate:    # 매칭 성공 → 좌표 EMA + count++
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

    def normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    # -------------------- 4) 축 추정 --------------------

    def update_cone_array(self, pts: np.ndarray, mode: str = "rail"):
        if pts.shape[0] == 0:
            self.sorted_pts = np.empty((0, 2), dtype=float)
            return

        # --- 공통: PCA로 레일 축 추정 & EMA로 안정화 ---
        if pts.shape[0] >= 2:
            mu = pts.mean(axis=0)
            X  = pts - mu
            C  = X.T @ X
            eigvals, eigvecs = np.linalg.eigh(C)
            u = eigvecs[:, np.argmax(eigvals)]
            u = u / (np.linalg.norm(u) + 1e-12)

            # 축 부호 일관성
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

        # --- 정렬 ---
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
        rospy.loginfo(f"Parking slot found!!!!!!")
        return int(k)

    # -------------------- 6) 좌표 생성 --------------------

    def generate_parking_coords(self):
        gap_idx = self.find_parking_slot(self.sorted_pts)
        if gap_idx is not None:
            ref  = self.sorted_pts[gap_idx]
            stop = self.sorted_pts[gap_idx + 1]

            # ref -> stop 방향의 벡터
            direction_vector = stop - ref
            # 우측 직각 벡터 계산
            right_vector = np.array([direction_vector[1], -direction_vector[0]], dtype=float)
            # 우측 벡터 정규화
            norm_right_vector = right_vector / (np.linalg.norm(right_vector) + 1e-12)
            # 주차 좌표 계산
            self.parking_coords = ref + self.parking_radius * norm_right_vector

            self.reference_coords = ref
            self.parking_stop_coords = stop

            self.parking_slot_found  = True
            self.parking_in_progress = True
        else:   
            self.parking_slot_found = False

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
        self.parking_path.poses.clear()

        now = rospy.Time.now()
        last_yaw = 0.0
        for i, (x, y) in enumerate(pts):
            if i < len(pts)-1:
                nx, ny = pts[i+1]
                yaw = math.atan2(ny - y, nx - x)
                last_yaw = yaw
            else:
                yaw = last_yaw

            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.frame_id
            pose_stamped.header.stamp = now
            pose_stamped.pose.position.x = float(x)
            pose_stamped.pose.position.y = float(y)
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw

            self.parking_path.poses.append(pose_stamped)


        if self.parking_in_progress and not self.parking_path_is_pub:
            self.parking_path_pub.publish(self.parking_path)
            self.parking_path_is_pub = True
            rospy.loginfo(f"Parking path published!!!!!!")
            self.publish_markers()

    # -------------------- rev 경로 생성 --------------------
    def generate_return_path(self):
        path = self.parking_path.poses
        if len(path) < 2:
            return
        pts = [(e.pose.position.x, e.pose.position.y) for e in path][::-1]

        new_poses = []
        now = rospy.Time.now()
        last_yaw = 0.0
        for i, (x, y) in enumerate(pts):
            if i < len(pts) - 1:
                nx, ny = pts[i+1]
                yaw = math.atan2(ny - y, nx - x)
                last_yaw = yaw
            else:
                yaw = last_yaw
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
            
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.frame_id
            pose_stamped.header.stamp = now
            pose_stamped.pose.position.x = float(x)
            pose_stamped.pose.position.y = float(y)
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw
            new_poses.append(pose_stamped)

        self.return_path.poses = new_poses
        self.return_path.header.stamp = rospy.Time.now()
        self.return_path_pub.publish(self.return_path)



    # -------------------- raw path 생성 --------------------
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
            if self.parking_done and len(self.parking_path.poses) >= 2:
                self.generate_return_path()
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

        dist_to_target = np.hypot(self.curr_pose.position.x - parking_stop_x,\
                                   self.curr_pose.position.y - parking_stop_y)
        
        if not self.parking_done_once:
            if dist_to_target < self.Parking_Stop_Radius and self.parking_in_progress:
                self.parking_stop_done = True
                self.parking_done_once = True
                rospy.loginfo(f"Parking Stop Done!!!!!!!!!")

        self.parking_stop_done_pub.publish(1 if self.parking_stop_done else 0)

    def parking_done_confirm(self):
        path_list = self.parking_path.poses
        if self.curr_pose is None or not path_list:
            return

        last_ps = path_list[-1].pose
        end_x = float(last_ps.position.x)
        end_y = float(last_ps.position.y)
        px, py = float(self.curr_pose.position.x), float(self.curr_pose.position.y)

        self.parking_done = (math.hypot(px - end_x, py - end_y) <= self.Parking_Success_Radius)
        self.parking_done_pub.publish(1 if self.parking_done else 0)

    def return_done_confirm(self):
        path_list = self.parking_path.poses
        if self.curr_pose is None or not path_list:
            return

        last_ps = path_list[-1].pose
        end_x = float(last_ps.position.x)
        end_y = float(last_ps.position.y)
        px, py = float(self.curr_pose.position.x), float(self.curr_pose.position.y)

        if math.hypot(px - end_x, py - end_y) <= self.Parking_Success_Radius:
            self.parking_end = True
            self.parking_end_pub.publish(1 if self.parking_end else 0)
            self.parking_in_progress = False
            self.return_phase = 3


    # -------------------- 마커 발행 ---------------------
    
    def publish_markers(self):
        """
        RViz 시각화 함수 (delivery_planner 스타일로 안정성 및 가독성 개선)
        - SPHERE: reference_coords(파랑), parking_stop_coords(빨강), parking_coords(초록)
        - CYLINDER: Parking_Stop_Radius(노랑), Parking_Success_Radius(보라)
        - LINE_STRIP: parking_path(흰색)
        """
        stamp = rospy.Time.now()
        frame = self.frame_id

        # --- Marker 1: reference_coords (파란색 구) ---
        m1 = Marker()
        m1.header.frame_id = frame
        m1.header.stamp = stamp
        m1.ns = "parking_points"
        m1.id = 0
        m1.type = Marker.SPHERE
        m1.action = Marker.ADD
        if self.reference_coords is not None:
            m1.pose.position.x = self.reference_coords[0] - self.rviz_offset_x
            m1.pose.position.y = self.reference_coords[1] - self.rviz_offset_y
            m1.pose.orientation.w = 1.0
            m1.scale.x = m1.scale.y = m1.scale.z = 0.4
            m1.color.r, m1.color.g, m1.color.b, m1.color.a = (0.2, 0.4, 1.0, 0.9) # 파란색
        else:
            m1.action = Marker.DELETE # 데이터가 없으면 마커 삭제
        self.marker_pub.publish(m1)

        # --- Marker 2: parking_stop_coords (빨간색 구) ---
        m2 = Marker()
        m2.header.frame_id = frame
        m2.header.stamp = stamp
        m2.ns = "parking_points"
        m2.id = 1
        m2.type = Marker.SPHERE
        m2.action = Marker.ADD
        if self.parking_stop_coords is not None:
            m2.pose.position.x = self.parking_stop_coords[0] - self.rviz_offset_x
            m2.pose.position.y = self.parking_stop_coords[1] - self.rviz_offset_y
            m2.pose.orientation.w = 1.0
            m2.scale.x = m2.scale.y = m2.scale.z = 0.4
            m2.color.r, m2.color.g, m2.color.b, m2.color.a = (1.0, 0.2, 0.2, 0.9) # 빨간색
        else:
            m2.action = Marker.DELETE
        self.marker_pub.publish(m2)

        # --- Marker 3: parking_coords (초록색 구) ---
        m3 = Marker()
        m3.header.frame_id = frame
        m3.header.stamp = stamp
        m3.ns = "parking_points"
        m3.id = 2
        m3.type = Marker.SPHERE
        m3.action = Marker.ADD
        if self.parking_coords is not None:
            m3.pose.position.x = self.parking_coords[0] - self.rviz_offset_x
            m3.pose.position.y = self.parking_coords[1] - self.rviz_offset_y
            m3.pose.orientation.w = 1.0
            m3.scale.x = m3.scale.y = m3.scale.z = 0.4
            m3.color.r, m3.color.g, m3.color.b, m3.color.a = (0.2, 1.0, 0.3, 0.9) # 초록색
        else:
            m3.action = Marker.DELETE
        self.marker_pub.publish(m3)

        # --- Marker 4: Parking_Stop_Radius (노란색 원) ---
        m4 = Marker()
        m4.header.frame_id = frame
        m4.header.stamp = stamp
        m4.ns = "parking_radii"
        m4.id = 10
        m4.type = Marker.CYLINDER
        m4.action = Marker.ADD
        if self.parking_stop_coords is not None:
            m4.pose.position.x = self.parking_stop_coords[0] - self.rviz_offset_x
            m4.pose.position.y = self.parking_stop_coords[1] - self.rviz_offset_y
            m4.pose.orientation.w = 1.0
            m4.scale.x = m4.scale.y = self.Parking_Stop_Radius * 2.0
            m4.scale.z = 0.1
            m4.color.r, m4.color.g, m4.color.b, m4.color.a = (1.0, 1.0, 0.0, 0.35) # 노란색 (반투명)
        else:
            m4.action = Marker.DELETE
        self.marker_pub.publish(m4)

        # --- Marker 5: Parking_Success_Radius (보라색 원) ---
        m5 = Marker()
        m5.header.frame_id = frame
        m5.header.stamp = stamp
        m5.ns = "parking_radii"
        m5.id = 11
        m5.type = Marker.CYLINDER
        m5.action = Marker.ADD
        
        success_center = None
        if hasattr(self, "parking_path") and self.parking_path.poses:
            last_pose = self.parking_path.poses[-1].pose.position
            success_center = (last_pose.x, last_pose.y)
        elif self.parking_coords is not None:
            success_center = self.parking_coords
            
        if success_center is not None:
            m5.pose.position.x = success_center[0] - self.rviz_offset_x
            m5.pose.position.y = success_center[1] - self.rviz_offset_y
            m5.pose.orientation.w = 1.0
            m5.scale.x = m5.scale.y = self.Parking_Success_Radius * 2.0
            m5.scale.z = 0.1
            m5.color.r, m5.color.g, m5.color.b, m5.color.a = (0.6, 0.2, 0.8, 0.35) # 보라색 (반투명)
        else:
            m5.action = Marker.DELETE
        self.marker_pub.publish(m5)

        # --- Marker 6: Parking Path (흰색 선) ---
        m6 = Marker()
        m6.header.frame_id = frame
        m6.header.stamp = stamp
        m6.ns = "parking_path_line"
        m6.id = 20
        m6.type = Marker.LINE_STRIP
        m6.action = Marker.ADD
        if hasattr(self, "parking_path") and len(self.parking_path.poses) >= 2:
            m6.scale.x = 0.1 # 선의 두께
            m6.color.r, m6.color.g, m6.color.b, m6.color.a = (1.0, 1.0, 1.0, 1.0) # 흰색
            
            for pose_stamped in self.parking_path.poses:
                pt = Point()
                pt.x = pose_stamped.pose.position.x - self.rviz_offset_x
                pt.y = pose_stamped.pose.position.y - self.rviz_offset_y
                m6.points.append(pt)
        else:
            m6.action = Marker.DELETE
        self.marker_pub.publish(m6)