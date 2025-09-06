import math
import numpy as np
import rospy

from geometry_msgs.msg import Pose
from lidar_object_detection.msg import ObjectInfo
from std_msgs.msg import Int16
from tf.transformations import euler_from_quaternion

from erp_drive.msg import PathReference, PathReferenceElement


class ParkingPlanner:
    def __init__(self, rviz_offset_x: float, rviz_offset_y: float):
        # === RViz 보조 설정 ===
        self.rviz_offset_x = float(rviz_offset_x)
        self.rviz_offset_y = float(rviz_offset_y)

        # === 인지/매칭 파라미터 ===
        self.num_rows = 13
        self.ROI_radius = 1.5                # 차량 기준 ROI 반경 (m)
        self.gate = 0.30                     # 매칭 게이트 (m)
        self.Parking_Success_Radius = 0.5    # 성공 반경 (m)
        self.debounce_k = 3                  # 디바운스 임계치 (프레임 카운트)
        self.frame_id = "map"

        # === 주차 레일/슬롯 정의 ===
        self.parking_start_xy = np.array([935534.553, 1915840.552], dtype=float)
        self.parking_end_xy   = np.array([935537.158, 1915844.797], dtype=float)
        self.slots = {
            1: {"stop_xy": (935534.553, 1915840.552), "parking_xy": (935537.841, 1915840.430)},
            2: {"stop_xy": (935535.852, 1915842.632), "parking_xy": (935539.031, 1915842.589)},
            3: {"stop_xy": (935537.158, 1915844.797), "parking_xy": (935540.203, 1915844.801)},
        }

        # === 예상 라바콘 점/관측 카운터 ===
        self.obstacles = self.expected_pts(self.parking_start_xy, self.parking_end_xy, self.num_rows)
        self.vis_cnt = np.zeros(self.num_rows, dtype=np.int16)     # 관측 누적치
        self.visited = np.zeros(self.num_rows, dtype=np.int16)     # 확정(≥debounce_k)

        # === 상태 변수 ===
        self.curr_pose = None
        self.stop_xy = None
        self.confirmed_slot = -1
        self.return_phase = 0

        # 미션 상태
        self.parking_in_progress = False    # 주차 미션 진행 플래그
        self.parking_stop_done   = False    # 1차 정지 완료
        self.parking_done_once   = False    # 1차 정지 완료 전이 감지용
        self.parking_done        = False    # 전진 경로 끝점 도달
        self.return_in_progress  = False    # 역방향 복귀 진행 중
        self.return_done         = False    # 복귀 완료
        self.flip_arccos         = False    # arccos 경로 뒤집기 스위치

        # === 경로/타이머/플래그 ===
        self.resolution = 0.3               # 경로 샘플 간격 (m)
        self.timer = None                   # 경로 생성 지연 타이머(oneshot)
        self.parking_path_is_pub = False    # 경로 1회 퍼블 방지

        # 역방향(복귀) 스케줄링 플래그/타이머
        self._reverse_timer = None
        self._reverse_scheduled = False
        self.reverse_path_published = False
        self._last_parking_done_pub = 0

        # === PathReference 메시지 버퍼 (타입 혼용 방지) ===
        self.parking_path = PathReference()
        try:
            self.parking_path.header.frame_id = self.frame_id
        except Exception:
            pass
        try:
            # 메시지 컨테이너 비워두기
            self.parking_path.path.clear()
        except Exception:
            self.parking_path.path = []

        # === ROS Pub/Sub (퍼블리셔 먼저 생성) ===
        self.parking_stop_done_pub = rospy.Publisher("/parking_stop_done", Int16, queue_size=1)
        self.parking_path_pub      = rospy.Publisher("/parking_path",      PathReference, queue_size=1)
        self.parking_done_pub      = rospy.Publisher("/parking_done",      Int16,         queue_size=1)
        self.parking_end_pub       = rospy.Publisher("/parking_end",       Int16,         queue_size=1)

        rospy.Subscriber("/obstacle_info", ObjectInfo, self.obstacle_info_CB, queue_size=1)
        rospy.Subscriber("/current_pose",  Pose,       self.curr_pose_CB,     queue_size=1)

    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg
        if not self.parking_stop_done:
            self.parking_stop_confirm()
        if self.parking_in_progress:
            self.return_to_gpath()

    
    # 라바콘 예상 좌표 생성
    def expected_pts(self, parking_start_xy, parking_end_xy, num_rows):
        start = np.asarray(parking_start_xy, dtype=float).reshape(1, 2)
        end = np.asarray(parking_end_xy, dtype=float).reshape(1, 2)

        if num_rows <= 1:
            return start.copy()
        
        alphas = np.linspace(0.0, 1.0, num_rows, dtype=float).reshape(-1, 1)
        pts = start + alphas * (end - start)

        return pts
    
    # 라바콘 인지
    def obstacle_info_CB(self, msg: ObjectInfo):
        if self.curr_pose is None:
            return

        N = int(msg.objectCounts)
        if N <= 0:
            return

        cx = np.asarray(msg.centerX[:N], dtype=float)
        cy = np.asarray(msg.centerY[:N], dtype=float)

        valid = np.isfinite(cx) & np.isfinite(cy)
        if not valid.any():
            return
        cx, cy = cx[valid], cy[valid]

        d2_local = cx*cx + cy*cy
        roi_mask = d2_local <= self.ROI_radius ** 2
        if not roi_mask.any():
            return
        cx, cy = cx[roi_mask], cy[roi_mask]

        x_r, y_r = self.curr_pose.position.x, self.curr_pose.position.y
        q = self.curr_pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        c, s = math.cos(yaw), math.sin(yaw)

        x_g = x_r + (cx*c - cy*s)
        y_g = y_r + (cx*s + cy*c)
        obs_pts = np.stack((x_g, y_g), axis=1)

        exp = np.asarray(self.obstacles, dtype=float).reshape(-1, 2)
        if exp.size == 0 or obs_pts.size == 0:
            return

        diff = exp[:, None, :] - obs_pts[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)

        if self.vis_cnt.shape[0] != self.num_rows:
            self.vis_cnt = np.zeros(self.num_rows, dtype=np.int16)

        used_obs = np.zeros(obs_pts.shape[0], dtype=bool)
        matched_row = np.zeros(exp.shape[0], dtype=bool)

        flat_order = np.argsort(d2, axis=None)
        W = d2.shape[1]

        for f in flat_order:
            i, j = divmod(f, W)
            if matched_row[i] or used_obs[j]:
                continue
            if d2[i, j] <= self.gate ** 2:
                matched_row[i] = True
                used_obs[j] = True
                self.vis_cnt[i] = min(self.vis_cnt[i] + 1, 255)

        decay_mask = ~matched_row
        if decay_mask.any():
            dec_vals = self.vis_cnt[decay_mask]
            self.vis_cnt[decay_mask] = np.where(dec_vals > 0, dec_vals - 1, 0)

        self.visited[:] = (self.vis_cnt >= self.debounce_k).astype(np.uint8)

        if int(self.visited.sum()) >= 2:
            self.update_parking_status()


    # 주차 공간 상태 update
    def update_parking_status(self):
        if self.confirmed_slot != -1:
            return
        if not np.any(self.visited):
            return

        # 가장 나중에 들어왔던 라바콘 이전까지만 확인
        latest = self.visited.nonzero()[0].max()
        bounds = [0, 4, 8, 12]      # 주차칸 시작/끝 index
        empty_slot = -1

        # 사이 공간 인지된 라바콘 개수 확인
        for k in range(3):
            L, R = bounds[k], bounds[k+1]
            a, b = L+1, R
            if latest < (b-1):
                continue
            if np.sum(self.visited[a:b]) == 0:
                empty_slot = k + 1
                break

        # 주차 공간 확정 X
        if empty_slot == -1:
            return

        # 주차 공간 확정
        self.confirmed_slot = empty_slot

        # 주차 공간 확정되면 정지 위치 저장
        if empty_slot in self.slots:
            self.stop_xy = np.array(self.slots[empty_slot]["stop_xy"], dtype=float)
            self.parking_in_progress = True
        else:
            self.stop_xy = None

    # 정지 구현을 위한 pub
    def parking_stop_confirm(self):
        if self.stop_xy is None or self.curr_pose is None:
            self.parking_stop_done = False
            return
        
        stop_x, stop_y = self.stop_xy

        dist_to_target = np.hypot(self.curr_pose.position.x - stop_x,\
                                   self.curr_pose.position.y - stop_y)
        
        if not self.parking_done_once:
            if dist_to_target < self.Parking_Success_Radius and self.parking_in_progress:
                self.parking_stop_done = True
                self.parking_done_once = True

        # parking_stop_done = 1이 되면 controller_node에서 정지 후 path 받아서 local_path 따라 pure_pursuit
        self.parking_stop_done_pub.publish(1 if self.parking_stop_done else 0)
        self.delay_path_maker()

    # delay 후에 경로 생성
    def delay_path_maker(self):
        if self.timer is not None:
            self.timer.shutdown()

        self.timer = rospy.Timer(rospy.Duration(0.5), self.make_parking_path, oneshot=True)
        return
    
    # parking_path 생성
    def make_parking_path(self, event=None):
        if self.stop_xy is None or self.curr_pose is None or self.confirmed_slot not in self.slots:
            return

        start_xy = np.array(self.slots[self.confirmed_slot]["stop_xy"], dtype=float)
        goal_xy  = np.array(self.slots[self.confirmed_slot]["parking_xy"], dtype=float)

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

    # 현 위치 -> 주차 최종 좌표로 arccos 함수 생성
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
    
    # 해상도 0.3으로 spacing
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
    
    # 주차 후 g_path로 복귀
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

    # 주차 완료 pub
    def parking_done_confirm(self):
        path_list = self.parking_path.path

        if self.curr_pose is None or not path_list:
            return
        
        pose = getattr(path_list[-1].pose, "pose", path_list[-1].pose)
        end_x, end_y = float(pose.position.x), float(pose.position.y)
        px, py = float(self.curr_pose.position.x), float(self.curr_pose.position.y)

        self.parking_done = (math.hypot(px - end_x, py - end_y) <= self.Parking_Success_Radius)
        self.parking_done_pub.publish(1 if self.parking_done else 0)

    # g_path 복귀 pub
    def return_done_confirm(self):
        path_list = self.parking_path.path
        if self.curr_pose is None or not path_list:
            return
        pose = getattr(path_list[-1].pose, "pose", path_list[-1].pose)
        end_x, end_y = float(pose.position.x), float(pose.position.y)
        px, py = float(self.curr_pose.position.x), float(self.curr_pose.position.y)

        if math.hypot(px - end_x, py - end_y) <= self.Parking_Success_Radius:
            self.parking_end_pub.publish(1)
            self.parking_in_progress = False
            self.return_phase = 3