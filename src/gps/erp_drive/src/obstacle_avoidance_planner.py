#! /usr/bin/env python3

# path 경로를 받기 위함
import sys

# 경로 관련 모듈
from nav_msgs.msg import Path
from erp_drive.msg import PathReference
from erp_drive.msg import PathReferenceElement

# Pose 등등 위치 관련 모듈
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped

# 오일러 <-> 쿼터니언 변환 관련 모듈
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline

# 라이다 관련 -> 패키지명 확인 후 수정 
from std_msgs.msg import Int16
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray 

from constant import Lane, const

from lidar_object_detection.msg import ObjectInfo

from frenet_frame import (
    compute_s_and_yaw,
    cartesian_to_frenet,
    generate_quintic_path,
    frenet_to_cartesian
)

# ros 관련 모듈
import rospy
import rospkg

class ObstacleAvoidancePlanner:
    def __init__(self, g_path: dict, rviz_offset_x: float, rviz_offset_y: float):

        self.decel_request_pub = rospy.Publisher('/deceleration_request', Int16, queue_size=1)
        
        #dh
        self.override_radius = 1.0
        self.last_ccw = 0
        ###
        self.rviz_offset_x, self.rviz_offset_y = rviz_offset_x, rviz_offset_y
     
        self.g_path = g_path
        
        self.g_curr_idx = 0
        self.curr_pose = Pose()
        self.curr_lane = Lane.ONE

        self.obst_x = 0
        self.obst_y = 0
        self.obst_len_x = 0
        self.obst_len_y = 0
        self.obst_len_z = 0
        self.obst_type = ''
        self.objectCounts = 0
        
        self.avoidance_path = PathReference()
        self.avoidance_path_pub = rospy.Publisher('/local_path', PathReference, queue_size=1)
        self.local_path_done_signal_pub = rospy.Publisher('/local_path_done', Int16, queue_size=1)

        self.rviz_avoidance_path = Path()
        self.rviz_avoidance_path.header.frame_id = 'map'
        self.rviz_avoidance_path_pub = rospy.Publisher('/rviz_l_path', Path, queue_size=1)
        self.target_marker_pub = rospy.Publisher('/avoidance_target_marker', Marker, queue_size=1)


        self.rviz_curr_waypoint = PointStamped()
        self.rviz_curr_waypoint.header.frame_id = 'map'
        self.rviz_curr_waypoint_pub = rospy.Publisher('/rviz_curr_waypoint', PointStamped, queue_size=1)

        self.blocked_idx = -1
        self.rel_obstalce_info = None 
        self.obstacle_info = None
        self.avoidance_in_progress = False
        self.is_local_waypoint_set = False
        
        self.target_x, self.target_y = None, None

        self.obstacle_makers_pub = rospy.Publisher('/obstacle_marker', MarkerArray, queue_size=1)
        
        self.obstacle_info_sub = rospy.Subscriber("/obstacle_info", ObjectInfo, self.obstacle_info_CB)
        self.big_info_sub = rospy.Subscriber("/big_info", ObjectInfo, self.big_info_CB)
  

    def update_curr_pose(self, curr_pose: Pose):
        self.curr_pose = curr_pose
        self.is_pose_set = True
    
    def update_g_curr_idx(self, g_curr_idx: int):
        self.g_curr_idx = g_curr_idx

        self.rviz_curr_waypoint.header.stamp = rospy.Time.now()
        self.rviz_curr_waypoint.point.x = self.g_path[self.curr_lane][self.g_curr_idx].pose.position.x - self.rviz_offset_x
        self.rviz_curr_waypoint.point.y = self.g_path[self.curr_lane][self.g_curr_idx].pose.position.y - self.rviz_offset_y
        self.rviz_curr_waypoint_pub.publish(self.rviz_curr_waypoint)
        
    def get_transfomation_matrix(self):
        orientation_list = [
            self.curr_pose.orientation.x, 
            self.curr_pose.orientation.y,
            self.curr_pose.orientation.z,
            self.curr_pose.orientation.w
        ]
        
        transition = np.array([
            [1., 0., 0., self.curr_pose.position.x], 
            [0., 1., 0., self.curr_pose.position.y],
            [0., 0., 1., self.curr_pose.position.z],
            [0., 0., 0., 1.]
        ])
        
        inverse_transition = np.array([
            [1., 0., 0., -self.curr_pose.position.x], 
            [0., 1., 0., -self.curr_pose.position.y], 
            [0., 0., 1., -self.curr_pose.position.z],
            [0., 0., 0., 1.]
        ])
        
        rotation = quaternion_matrix(orientation_list)
        
        mat_v2g = transition @ rotation
        mat_g2v = rotation.T @ inverse_transition
        
        _, _, curr_yaw = euler_from_quaternion(orientation_list)
        
        return mat_v2g, mat_g2v, curr_yaw
    

    def ready_to_override(self):
        if self.target_x is None or self.target_y is None:
            return False

        dist = np.hypot(self.curr_pose.position.x - self.target_x,
                        self.curr_pose.position.y - self.target_y)

        if dist < self.override_radius:
            rospy.loginfo("[INFO] 범위 안에 들어옴 (%.2f m 이내)" % dist)
            return True

        return False


    def obstacle_info_CB(self, msg : ObjectInfo):

        self.objectCounts = msg.objectCounts

        self.rel_obstacle_info = np.array([msg.centerX, msg.centerY, msg.centerZ, msg.lengthX, msg.lengthY, msg.lengthZ])
        self.rel_obstacle_info = self.rel_obstacle_info[:, :msg.objectCounts]
        self.rel_obstacle_info[0] +=1.2

        # GPS 수신기의 위치에 따라 값이 변동됨.
        # ERP 기준으로는 ~~ 약 0.2???? @@@ 내일 하기 전에 꼭 변경하기
        # self.rel_obstacle_info[1] -= 0.2

        angle_filter = np.abs(np.arctan2(self.rel_obstacle_info[1], self.rel_obstacle_info[0])) < np.pi / 9
        self.rel_obstacle_info = self.rel_obstacle_info[:, angle_filter]
        
        if len(self.rel_obstacle_info[0]) == 0:
            if self.avoidance_success_confirm():
                self.avoidance_in_progress = False
            return
       
        self.obst_x = self.rel_obstacle_info[0][0]
        self.obst_y = self.rel_obstacle_info[1][0]
        self.obst_len_x = self.rel_obstacle_info[-3][0]
        self.obst_len_y = self.rel_obstacle_info[-2][0]
        self.obst_len_z = self.rel_obstacle_info[-1][0]
    
        mat_v2g, mat_g2v, curr_yaw = self.get_transfomation_matrix()

        self.g_obstacle_info = np.copy(self.rel_obstacle_info)

        info = np.array([
                self.rel_obstacle_info[0],
                self.rel_obstacle_info[1],
                np.zeros(len(self.rel_obstacle_info[0])),
                np.ones(len(self.rel_obstacle_info[0]))
            ])
        
        self.g_obstacle_info[:2] = (mat_v2g @ info)[:2]
        ############################################################

        _marker_array = MarkerArray()
        for idx in range(len(self.rel_obstacle_info[0])):
            _marker = Marker()
            _marker.ns = 'hi'
            _marker.header.frame_id = 'map'
            _marker.header.stamp = rospy.Time.now()
            _marker.id = idx
            _marker.type = Marker.CUBE
            _marker.action = Marker.ADD

            _marker.pose.position.x = self.g_obstacle_info[0][idx] - self.rviz_offset_x
            _marker.pose.position.y = self.g_obstacle_info[1][idx] - self.rviz_offset_y
            _marker.pose.position.z = self.g_obstacle_info[2][idx]

            _marker.pose.orientation.x = self.curr_pose.orientation.x
            _marker.pose.orientation.y = self.curr_pose.orientation.y
            _marker.pose.orientation.z = self.curr_pose.orientation.z
            _marker.pose.orientation.w = self.curr_pose.orientation.w
            
            _marker.color.a = 0.5
            _marker.color.r = 1.0
            _marker.color.g = 1.0
            _marker.color.b = 1.0

            _marker.scale.x = self.rel_obstacle_info[-3][idx]
            _marker.scale.y = self.rel_obstacle_info[-2][idx]
            _marker.scale.z = self.rel_obstacle_info[-1][idx]

            _marker_array.markers.append(_marker)

        self.obstacle_makers_pub.publish(_marker_array)

        if self.avoidance_in_progress:
            if self.ready_to_override():
                is_blocked, is_big, ccw = self.is_obst_on_path()
                if is_blocked:
                    rospy.loginfo("[중단] 타겟 반경 내 도달 + 장애물 감지 → 로컬패스 덮어쓰기")
                    self.is_local_waypoint_set = False
                    self.avoidance_in_progress = False

                    mat_v2g, mat_g2v, curr_yaw = self.get_transfomation_matrix()
                    self.make_avoidance_path(is_big, ccw, mat_v2g, mat_g2v, curr_yaw)
                    return

            if self.avoidance_success_confirm():
                self.avoidance_in_progress = False

            return
    
        is_blocked, is_big, ccw = self.is_obst_on_path()
        if not is_blocked:
            return

        self.make_avoidance_path(is_big, ccw, mat_v2g, mat_g2v, curr_yaw)

    def big_info_CB(self, msg : ObjectInfo):

        self.objectCounts = msg.objectCounts

        self.rel_obstacle_info = np.array([msg.centerX, msg.centerY, msg.centerZ, msg.lengthX, msg.lengthY, msg.lengthZ])
        self.rel_obstacle_info = self.rel_obstacle_info[:, :msg.objectCounts]
        self.rel_obstacle_info[0] += 1.2

        angle_filter = np.abs(np.arctan2(self.rel_obstacle_info[1], self.rel_obstacle_info[0])) < np.pi / 9
        self.rel_obstacle_info = self.rel_obstacle_info[:, angle_filter]
        
        if len(self.rel_obstacle_info[0]) == 0:
            if self.avoidance_success_confirm():
                self.avoidance_in_progress = False
            return
       
        self.obst_x = self.rel_obstacle_info[0][0]
        self.obst_y = self.rel_obstacle_info[1][0]
        self.obst_len_x = self.rel_obstacle_info[-3][0]
        self.obst_len_y = self.rel_obstacle_info[-2][0]
        self.obst_len_z = self.rel_obstacle_info[-1][0]
    
        mat_v2g, mat_g2v, curr_yaw = self.get_transfomation_matrix()

        self.g_obstacle_info = np.copy(self.rel_obstacle_info)

        info = np.array([
                self.rel_obstacle_info[0],
                self.rel_obstacle_info[1],
                np.zeros(len(self.rel_obstacle_info[0])),
                np.ones(len(self.rel_obstacle_info[0]))
            ])
        
        self.g_obstacle_info[:2] = (mat_v2g @ info)[:2]
        ############################################################

        _marker_array = MarkerArray()
        for idx in range(len(self.rel_obstacle_info[0])):
            _marker = Marker()
            _marker.ns = 'hi'
            _marker.header.frame_id = 'map'
            _marker.header.stamp = rospy.Time.now()
            _marker.id = idx
            _marker.type = Marker.CUBE
            _marker.action = Marker.ADD

            _marker.pose.position.x = self.g_obstacle_info[0][idx] - self.rviz_offset_x
            _marker.pose.position.y = self.g_obstacle_info[1][idx] - self.rviz_offset_y
            _marker.pose.position.z = self.g_obstacle_info[2][idx]

            _marker.pose.orientation.x = self.curr_pose.orientation.x
            _marker.pose.orientation.y = self.curr_pose.orientation.y
            _marker.pose.orientation.z = self.curr_pose.orientation.z
            _marker.pose.orientation.w = self.curr_pose.orientation.w
            
            _marker.color.a = 0.5
            _marker.color.r = 1.0
            _marker.color.g = 1.0
            _marker.color.b = 1.0

            _marker.scale.x = self.rel_obstacle_info[-3][idx]
            _marker.scale.y = self.rel_obstacle_info[-2][idx]
            _marker.scale.z = self.rel_obstacle_info[-1][idx]

            _marker_array.markers.append(_marker)

        self.obstacle_makers_pub.publish(_marker_array)

        # # 이미 회피 경로가 생성되어 진행 중인 경우
        # if self.avoidance_in_progress:
        #     if self.avoidance_success_confirm():
        #         self.avoidance_in_progress = False 
                
        #     return
        #dh
        if self.avoidance_in_progress:
            if self.ready_to_override():
                is_blocked, is_big, ccw = self.is_obst_on_path()
                if is_blocked:
                    rospy.loginfo("[중단] 타겟 반경 내 도달 + 장애물 감지 → 로컬패스 덮어쓰기")
                    self.is_local_waypoint_set = False
                    self.avoidance_in_progress = False

                    mat_v2g, mat_g2v, curr_yaw = self.get_transfomation_matrix()
                    self.make_avoidance_path(is_big, ccw, mat_v2g, mat_g2v, curr_yaw)
                    return

            if self.avoidance_success_confirm():
                self.avoidance_in_progress = False

            return
      
        is_blocked, is_big, ccw = self.is_obst_on_path()
        if not is_blocked:
            return

        self.make_avoidance_path(is_big, ccw, mat_v2g, mat_g2v, curr_yaw)

    

    def avoidance_success_confirm(self):
        if self.target_x is None and self.target_y is None:
            return False

        target_x = self.target_x
        target_y = self.target_y

        dist_to_target = np.hypot(
            self.curr_pose.position.x - target_x,
            self.curr_pose.position.y - target_y
        )

        rospy.loginfo(f"[Confirm] Dist to Target: {dist_to_target:.2f}m / Success Radius: {const.AVOIDANCE_SUCCESS_RAIDUS}m")

        if dist_to_target > const.AVOIDANCE_SUCCESS_RAIDUS:
            return False
        
        self.decel_request_pub.publish(Int16(data=0))
        
        signal = Int16(data=1)
        self.local_path_done_signal_pub.publish(signal)
        rospy.loginfo("성공")
        self.is_local_waypoint_set = False
        self.target_x, self.target_y = None, None

        return True

    def is_obst_on_path (self):
        # 인덱스 구간에 따라 감지 거리 설정
        if 850 <= self.g_curr_idx <= 1000:
            # 대형 장애물 구간
            roi_ld = 10.0
        elif 3160 <= self.g_curr_idx <= 3470:
            # 소형 장애물 구간
            roi_ld = 6.0
        else:
            # 기본 구간
            roi_ld = 10.0

        vehicle_radius = 0.6

        start = self.g_curr_idx + 3
        end = min(self.g_curr_idx + int(roi_ld // 0.2), len(self.g_path[self.curr_lane]))
        
        blocked_left = False
        blocked_mid = False
        blocked_right = False

        blocked_left_idx = 0
        blocked_mid_idx = 0
        blocked_right_idx = 0

        calc_info = np.copy(self.g_obstacle_info)

        for i in range(start, end):
            x = self.g_path[self.curr_lane][i].pose.position.x 
            y = self.g_path[self.curr_lane][i].pose.position.y

            quat = self.g_path[self.curr_lane][i].pose.orientation
            orientation_list = [quat.x, quat.y, quat.z, quat.w]
            _, _, yaw = euler_from_quaternion(orientation_list)

            # print('small width : ', const.LANE_WIDTH_SMALL)

            lx = self.g_path[self.curr_lane][i].pose.position.x - const.LANE_WIDTH_SMALL * np.cos(yaw + np.pi / 2)
            ly = self.g_path[self.curr_lane][i].pose.position.y - const.LANE_WIDTH_SMALL * np.sin(yaw + np.pi / 2)
            
            rx = self.g_path[self.curr_lane][i].pose.position.x + const.LANE_WIDTH_SMALL * np.cos(yaw + np.pi / 2)
            ry = self.g_path[self.curr_lane][i].pose.position.y + const.LANE_WIDTH_SMALL * np.sin(yaw + np.pi / 2)

            dist1 = np.hypot(calc_info[0] - lx, calc_info[1] - ly)
            dist2 = np.hypot(calc_info[0] - x,  calc_info[1] - y)
            dist3 = np.hypot(calc_info[0] - rx, calc_info[1] - ry)
            
            obst_radius = calc_info[-2] / 2
            # 원이 아니고 사각 범위 검사가 맞긴 한데..

            if not blocked_left and (dist1 <= obst_radius + vehicle_radius).any():
                # rospy.loginfo(f'\ndist1 : {dist1} \n dist2 : {dist2} \n dist3 : {dist3} \n obst_radius + vehicle_radius : {obst_radius + vehicle_radius}')
                blocked_left = True
                blocked_left_idx = i
                # rospy.loginfo(f'blocked_left_idx : {blocked_left_idx}')

            if not blocked_mid and (dist2 <= obst_radius + vehicle_radius).any():
                # rospy.loginfo(f'\ndist1 : {dist1} \n dist2 : {dist2} \n dist3 : {dist3} \n obst_radius + vehicle_radius : {obst_radius + vehicle_radius}')
                blocked_mid = True
                blocked_mid_idx = i
                # rospy.loginfo(f'blocked_mid_idx : {blocked_mid_idx}')

            if not blocked_right and (dist3 <= obst_radius + vehicle_radius).any():
                # rospy.loginfo(f'\ndist1 : {dist1} \n dist2 : {dist2} \n dist3 : {dist3} \n obst_radius + vehicle_radius : {obst_radius + vehicle_radius}')
                blocked_right = True
                blocked_right_idx = i
                # rospy.loginfo(f'blocked_right_idx : {blocked_right_idx}')

        # rospy.loginfo("calc done, let's compare")

        # mid만 걸린 경우도 상정해야 함.
        # 이 경우에는 CCW 활용하기?

        # 대형인 경우
        if blocked_left and blocked_right:
            rospy.loginfo('left mid right')
            self.blocked_idx = min(blocked_left_idx, blocked_mid_idx, blocked_right_idx)
            self.blocked_idx += max(blocked_left_idx, blocked_mid_idx, blocked_right_idx)
            self.blocked_idx //= 2
            # self.blocked_idx += 2
            ccw = 1 if self.curr_lane == Lane.ONE else -1
            return True, True, ccw

        # 소형인 경우
        if blocked_left:
            rospy.loginfo('left mid')
            self.blocked_idx = blocked_left_idx
            return True, False, 1

        # 소형인 경우
        if blocked_right:
            rospy.loginfo('mid right')
            self.blocked_idx = blocked_right_idx
            return True, False, -1
        
        self.blocked_idx = -1
        return False, False, 0

    # # avoidance_frenet_example.py (updated from make_avoidance_path)

    def make_avoidance_path(self, is_big: bool, ccw, mat_v2g: np.ndarray, mat_g2v: np.ndarray, curr_yaw: float):
    # 1. 현재 차선을 '이번 회피 기동의 기준 경로'로 명확하게 지정합니다.
        original_ref_path = self.g_path[self.curr_lane]

        # 2. '기준 경로'를 바탕으로 Frenet 좌표계 관련 값들을 계산합니다.
        s_list, yaw_list = compute_s_and_yaw(original_ref_path)

        rospy.loginfo(f"[회피 경로 생성] ccw: {ccw}, is_big: {is_big}, blocked_idx: {self.blocked_idx}")

        # 3. 차량의 현재 위치도 '기준 경로'에 맞춰 변환합니다.
        curr_x = self.curr_pose.position.x
        curr_y = self.curr_pose.position.y
        s0, l0 = cartesian_to_frenet(curr_x, curr_y, original_ref_path, s_list)

        lane_offset = 0.82

        rospy.loginfo(f"[BIG??]: {is_big}")

        #  if 710 <= self.g_curr_idx <= 930 and is_big:

        if 850 <= self.g_curr_idx <= 1000 and is_big:
            # 목표 차선을 변경합니다.
            self.curr_lane = Lane.TWO if self.curr_lane == Lane.ONE else Lane.ONE
            self.avoidance_in_progress = True

            target_lookahead = 40 # 50 * 0.2m = 10m 앞을 목표로 설정 (튜닝 필요)
            target_idx = min(self.g_curr_idx + target_lookahead, len(self.g_path[self.curr_lane]) - 1)
            
            target_pose = self.g_path[self.curr_lane][target_idx].pose
            self.target_x = target_pose.position.x
            self.target_y = target_pose.position.y

            self.decel_request_pub.publish(Int16(data=1))

            rospy.loginfo(f"단순 차선 변경 완료 목표 지점 설정: ({self.target_x:.2f}, {self.target_y:.2f})")

            target_marker = Marker()
            target_marker.header.frame_id = 'map'
            target_marker.header.stamp = rospy.Time.now()
            target_marker.ns = 'virtual_target' # 네임스페이스를 다르게 하여 기존 마커와 구분
            target_marker.id = 0
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD

            target_marker.pose.position.x = self.target_x - self.rviz_offset_x
            target_marker.pose.position.y = self.target_y - self.rviz_offset_y
            target_marker.pose.position.z = 0.5 # 살짝 띄워서 잘 보이게 함

            target_marker.pose.orientation.w = 1.0

            target_marker.scale.x = 0.8
            target_marker.scale.y = 0.8
            target_marker.scale.z = 0.8

            target_marker.color.a = 0.8
            target_marker.color.r = 0.0 # 파란색
            target_marker.color.g = 0.5
            target_marker.color.b = 1.0

            self.target_marker_pub.publish(target_marker)

            return
        
        elif 3160 <= self.g_curr_idx <= 3260:
            # 목표 차선 변경 없이, 현재 차선 내에서 회피합니다.
            # self.curr_lane은 변경하지 않습니다.
            pass # 특별히 할 작업 없음
        
        # [조건] 회피 구간 밖 - 회피하지 않음
        else:
            rospy.loginfo("회피 구간 밖 - 장애물 감지되었지만 회피하지 않음")
            return
        
        # 회피 목표 지점 계산
        blend_margin = 0
        target_idx = self.blocked_idx + blend_margin
        s1 = s_list[target_idx]
        l1 = ccw * lane_offset

        # 5차 다항식 경로 생성
        l_path = generate_quintic_path(s0, l0, s1, l1)

        resolution = int((s1 - s0) / 0.2)
        self.avoidance_path = PathReference()
        self.rviz_avoidance_path.poses = []

        for s in np.linspace(s0, s1, resolution):
            l = l_path(s)
            
            # 5. 좌표를 되돌릴 때도 일관성 있게 맨 처음 지정한 '기준 경로'를 사용합니다.
            x, y, yaw = frenet_to_cartesian(s, l, original_ref_path, s_list, yaw_list)

            avoidance_path_element = PathReferenceElement()
            avoidance_path_element.pose.position.x = x
            avoidance_path_element.pose.position.y = y
            avoidance_path_element.pose.position.z = 0.

            quat = quaternion_from_euler(0., 0., yaw)
            avoidance_path_element.pose.orientation.x = quat[0]
            avoidance_path_element.pose.orientation.y = quat[1]
            avoidance_path_element.pose.orientation.z = quat[2]
            avoidance_path_element.pose.orientation.w = quat[3]

            avoidance_path_element.mode = 11
            self.avoidance_path.path.append(avoidance_path_element)

            rviz_local_pose = PoseStamped()
            rviz_local_pose.header.frame_id = 'map'
            rviz_local_pose.header.stamp = rospy.Time.now()

            rviz_local_pose.pose.position.x = x - self.rviz_offset_x
            rviz_local_pose.pose.position.y = y - self.rviz_offset_y
            rviz_local_pose.pose.position.z = 0.

            rviz_local_pose.pose.orientation.x = quat[0]
            rviz_local_pose.pose.orientation.y = quat[1]
            rviz_local_pose.pose.orientation.z = quat[2]
            rviz_local_pose.pose.orientation.w = quat[3]
            
            self.rviz_avoidance_path.poses.append(rviz_local_pose)

        # # ────────── ① [추가] 꼬리 세그먼트 붙이기 ──────────
        # sign = -np.sign(l1)            # l1>0(왼쪽) → 우회전(−10°), l1<0 → 좌회전(+10°)
        # if sign != 0:                  # 중앙이면 꼬리 생략
        #     self._append_tail(self.avoidance_path.path[-1], sign)

        self.target_x = self.avoidance_path.path[-1].pose.position.x
        self.target_y = self.avoidance_path.path[-1].pose.position.y

        target_marker = Marker()
        target_marker.header.frame_id = 'map'
        target_marker.header.stamp = rospy.Time.now()
        target_marker.ns = 'target'
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD

        target_marker.pose.position.x = self.avoidance_path.path[-1].pose.position.x - self.rviz_offset_x
        target_marker.pose.position.y = self.avoidance_path.path[-1].pose.position.y - self.rviz_offset_y
        target_marker.pose.position.z = 0.0

        target_marker.pose.orientation = self.avoidance_path.path[-1].pose.orientation

        target_marker.scale.x = 0.5  # 화살표 길이
        target_marker.scale.y = 0.5
        target_marker.scale.z = 0.5

        target_marker.color.a = 1.0
        target_marker.color.r = 1.0
        target_marker.color.g = 0.3
        target_marker.color.b = 0.0

        self.target_marker_pub.publish(target_marker)

        # self.target_x = self.avoidance_path.path[-1].pose.position.x
        # self.target_y = self.avoidance_path.path[-1].pose.position.y


       

        self.rviz_avoidance_path.header.stamp = rospy.Time.now()
        
        self.is_local_waypoint_set = True
        self.avoidance_in_progress = True
        

    def publish_avoidance_path(self):

        if self.is_local_waypoint_set:
            self.avoidance_path_pub.publish(self.avoidance_path)
            self.rviz_avoidance_path_pub.publish(self.rviz_avoidance_path)
        

    # def CCW(self, blocked_coord, obst_coord):
    #     x = self.g_path[self.curr_lane][self.g_curr_idx].pose.position.x
    #     y = self.g_path[self.curr_lane][self.g_curr_idx].pose.position.y
        
    #     cross_product = (blocked_coord[0] - x) * (obst_coord[1] - blocked_coord[1]) - (blocked_coord[1] - y) * (obst_coord[0] - blocked_coord[0])

    #     if cross_product > 0:
    #         return -1
        
    #     elif cross_product < 0:
    #         return 1
        
    #     return 0