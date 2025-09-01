import math
import rospy
import numpy as np

from std_msgs.msg import Int16
from geometry_msgs.msg import Pose
from lidar_object_detection.msg import ObjectInfo
from tf.transformations import euler_from_quaternion

from constant import const

class ParkingPlanner:
    def __init__(self, rviz_offset_x: float, rviz_offset_y: float):
        self.rviz_offset_x = rviz_offset_x
        self.rviz_offset_y = rviz_offset_y

        self.num_rows = 13
        self.ROI_radius   = 1.5
        self.gate         = 0.30
        self.frame_id     = "map"

        self.parking_start_xy   = np.array([935534.553, 1915840.552], dtype=float)
        self.parking_end_xy     = np.array([935537.158, 1915844.797], dtype=float)

        self.slots = {
            1: {"stop_xy": (935534.553, 1915840.552), "parking_xy": (935537.841, 1915840.430)},
            2: {"stop_xy": (935535.852, 1915842.632), "parking_xy": (935539.031, 1915842.589)},
            3: {"stop_xy": (935537.158, 1915844.797), "parking_xy": (935540.203, 1915844.801)}
        }

        self.slot_size = 3
        self.confirmed_slot = -1
        self.debounce_k = 3

        self.curr_idx = None
        self.curr_pose = None
        self.stop_xy = None
        self.obstacles = self.expected_pts(self.parking_start_xy, self.parking_end_xy, self.num_rows)
        self.vis_cnt = np.zeros(self.num_rows, dtype=np.int16)
        self.visited = np.zeros(self.num_rows, dtype=np.int16)

        self.parking_stop_done = False
        self.parking_done_once = False
        self.parking_in_progress = False

        # --- ROS Pub/Sub ---
        self.parking_stop_done_pub = rospy.Publisher("/parking_stop_done", Int16, queue_size=1)

        rospy.Subscriber("/obstacle_info", ObjectInfo, self.obstacle_info_CB, queue_size=1)
        rospy.Subscriber("/current_pose", Pose, self.curr_pose_CB, queue_size=1)
        rospy.Subscriber("/curr_idx", Int16, self.curr_idx_CB)


    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg

    def curr_idx_CB(self, msg: Int16):
        self.curr_idx = msg.data
        self.parking_stop_confirm()

    def expected_pts(self, parking_start_xy, parking_end_xy, num_rows):
        start = np.asarray(parking_start_xy, dtype=float).reshape(1, 2)
        end   = np.asarray(parking_end_xy,   dtype=float).reshape(1, 2)

        if num_rows <= 1:
            return start.copy()
        
        alphas = np.linspace(0.0, 1.0, num_rows, dtype=float).reshape(-1, 1)
        pts = start + alphas * (end - start)

        return pts
    
    def obstacle_info_CB(self, msg: ObjectInfo):
        if self.curr_pose is None:
            return
        
        N = int(msg.objectCounts)
        if N <= 0:
            return
        
        cx = np.asarray(msg.centerX[:N], dtype=float)
        cy = np.asarray(msg.centerY[:N], dtype=float)

        valid = np.isfinite(cx) & np.isfinite(cy)
        if not np.any(valid):
            return
        cx, cy = cx[valid], cy[valid]
        if cx.size == 0:
            return
        
        d2_local = cx * cx + cy * cy
        roi_mask = d2_local <= (self.ROI_radius ** 2)
        if not np.any(roi_mask):
            return
        cx, cy = cx[roi_mask], cy[roi_mask]
        if cx.size == 0:
            return

        x_r = self.curr_pose.position.x
        y_r = self.curr_pose.position.y
        q = self.curr_pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        c, s = math.cos(yaw), math.sin(yaw)

        x_g = x_r + (cx * c - cy * s)
        y_g = y_r + (cx * s + cy * c)
        obs_pts = np.stack((x_g, y_g), axis=1)

        exp = np.asarray(self.obstacles, dtype=float).reshape(-1, 2)
        if exp.size == 0 or obs_pts.size == 0:
            return
        
        diff = exp[:, None, :] - obs_pts[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        gate2 = float(self.gate) ** 2

        if not hasattr(self, "vis_cnt") or self.vis_cnt.shape[0] != self.num_rows:
            self.vis_cnt = np.zeros(self.num_rows, dtype=np.int16)
        if not hasattr(self, "debounce_k"):
            self.debounce_k = 3

        used_obs = np.zeros(obs_pts.shape[0], dtype=bool)
        matched_row = np.zeros(exp.shape[0], dtype=bool)

        flat_order = np.argsort(d2, axis=None)
        W = d2.shape[1]

        for f in flat_order:
            i = f // W
            j = f % W
            if matched_row[i] or used_obs[j]:
                continue
            if d2[i, j] <= gate2:
                matched_row[i] = True
                used_obs[j] = True
                self.vis_cnt[i] = self.vis_cnt[i] + 1
                if self.vis_cnt[i] > 255:
                    self.vis_cnt[i] = 255

        decay_mask = ~matched_row

        if np.any(decay_mask):
            dec_vals = self.vis_cnt[decay_mask]
            self.vis_cnt[decay_mask] = np.where(dec_vals > 0, dec_vals - 1, 0)

        self.visited[:] = (self.vis_cnt >= int(self.debounce_k)).astype(np.uint8)

        if int(np.sum(self.visited)) >= 2:
            self.update_parking_status()

    def update_parking_status(self):
        if self.confirmed_slot != -1:
            return
        if not np.any(self.visited):
            return

        latest = self.visited.nonzero()[0].max()
        bounds = [0, 4, 8, 12]
        empty_slot = -1

        for k in range(3):
            L, R = bounds[k], bounds[k+1]
            a, b = L+1, R
            if latest < (b-1):
                continue
            if np.sum(self.visited[a:b]) == 0:
                empty_slot = k + 1
                break

        if empty_slot == -1:
            return

        self.confirmed_slot = empty_slot

        if empty_slot in self.slots:
            self.stop_xy = np.array(self.slots[empty_slot]["stop_xy"], dtype=float)
            self.parking_in_progress = True
        else:
            self.stop_xy = None

    def parking_stop_confirm(self):
        if self.stop_xy is None or self.curr_pose is None:
            self.parking_stop_done = False
            return
        
        stop_x, stop_y = self.stop_xy

        dist_to_target = np.hypot(self.curr_pose.position.x - stop_x,\
                                   self.curr_pose.position.y - stop_y)
        
        if not self.parking_done_once:
            if dist_to_target < const.DELIVERY_SUCCESS_RADIUS and self.parking_in_progress:
                self.parking_stop_done = True
                self.parking_done_once = True

        self.parking_stop_done_pub.publish(1 if self.parking_stop_done else 0)
