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
        
    def obstacle_info_CB(self, msg : ObjectInfo):
        # if self.g_curr_idx >= 1240:
        #     return

        self.objectCounts = msg.objectCounts

        # msg에 내용물이 없는 경우
        # if msg.objectCounts == 0:
        #     rospy.loginfo('none')
        #     return
        
        ### lidar 좌표계 상의 정보를 global 좌표계 상 정보로 변경 ####
        self.rel_obstacle_info = np.array([msg.centerX, msg.centerY, msg.centerZ, msg.lengthX, msg.lengthY, msg.lengthZ])
        self.rel_obstacle_info = self.rel_obstacle_info[:, :msg.objectCounts]
        self.rel_obstacle_info[0] += 1.2
        
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

        # 이미 회피 경로가 생성되어 진행 중인 경우
        if self.avoidance_in_progress:
            if self.avoidance_success_confirm():
                self.avoidance_in_progress = False 
                
            return

        # 현재 주행 중인 global path 위에 장애물이 존재하지 않는 경우
        is_blocked, is_big, ccw = self.is_obst_on_path()
        if not is_blocked:
            return

        self.make_avoidance_path(is_big, ccw, mat_v2g, mat_g2v, curr_yaw)

    def avoidance_success_confirm(self):
        if self.target_x is None and self.target_y is None:
            return False
        
        target_x = self.target_x
        target_y = self.target_y

        dist_to_target = np.hypot(self.curr_pose.position.x - target_x, self.curr_pose.position.y - target_y)


        if dist_to_target > const.AVOIDANCE_SUCCESS_RAIDUS:
            return False
    
        signal = Int16(data=1)
        self.local_path_done_signal_pub.publish(signal)
        
        self.is_local_waypoint_set = False
        self.target_x, self.target_y = None, None
        
        return True

    def is_obst_on_path (self):
        roi_ld = 6.5
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
            return True, True, 0

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
    
    def setting_target_point(self, is_big: bool, ccw, mat_g2v: np.ndarray, curr_yaw: float):
        quat = self.g_path[self.curr_lane][self.blocked_idx].pose.orientation 
        orientation_list = [quat.x, quat.y, quat.z, quat.w]
        _, _, global_target_yaw = euler_from_quaternion(orientation_list)
        yaw = global_target_yaw - curr_yaw
        
        if 255 <= self.g_curr_idx <= 500:
            self.obst_type = 'small'   
            g_x = self.g_path[self.curr_lane][self.blocked_idx].pose.position.x
            g_y = self.g_path[self.curr_lane][self.blocked_idx].pose.position.y
            
            lane_width = const.LANE_WIDTH_SMALL
            # ccw = self.CCW((g_x, g_y), self.g_obstacle_info[:, 0])
            
            x, y, _, _ = mat_g2v @ np.array([g_x, g_y, 0., 1.]).T
            x += np.cos(yaw + np.pi / 2) * lane_width * ccw
            y += np.sin(yaw + np.pi / 2) * lane_width * ccw
        

        elif 845 <= self.g_curr_idx <= 1240:
            self.obst_type = 'big'   
            lane_width = const.LANE_WIDTH
            if self.curr_lane == Lane.ONE:
                ccw = -1
                self.curr_lane = Lane.TWO

            else: 
                ccw = 1
                self.curr_lane = Lane.ONE
        
            g_x = self.g_path[self.curr_lane][self.blocked_idx].pose.position.x
            g_y = self.g_path[self.curr_lane][self.blocked_idx].pose.position.y
            
            x, y, _, _ = mat_g2v @ np.array([g_x, g_y, 0., 1.]).T
        
        else:
            x, y, _, _ = mat_g2v @ np.array([self.curr_pose.position.x, self.curr_pose.position.y, 0, 1]).T

        return x, y, yaw
    

    # # avoidance_frenet_example.py (updated from make_avoidance_path)

    def make_avoidance_path(self, is_big: bool, ccw, mat_v2g: np.ndarray, mat_g2v: np.ndarray, curr_yaw: float):
        s_list, yaw_list = compute_s_and_yaw(self.g_path[self.curr_lane])

        rospy.loginfo(f"[회피 경로 생성] ccw: {ccw}, is_big: {is_big}, blocked_idx: {self.blocked_idx}")


        curr_x = self.curr_pose.position.x
        curr_y = self.curr_pose.position.y
        s0, l0 = cartesian_to_frenet(curr_x, curr_y, self.g_path[self.curr_lane], s_list)
        #여기 만져봄
        lane_offset = 0.75 #0.75
        s1 = s_list[self.blocked_idx]#-0.5 # 회피 경로 끝 지점 
        #s1 = s0+3.0
        l1 = ccw * lane_offset

        l_path = generate_quintic_path(s0, l0, s1, l1)
        #l_path = generate_quintic_path(s0, l0, s1, l1, v0=0.0, v1=1.0, a0=0.0, a1=0.0)

        resolution = int((s1 - s0) / 0.2)
        self.avoidance_path = PathReference()
        self.rviz_avoidance_path.poses = []

        for s in np.linspace(s0, s1, resolution):
            l = l_path(s)
            x, y, yaw = frenet_to_cartesian(s, l, self.g_path[self.curr_lane], s_list, yaw_list)

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

        self.target_x = self.avoidance_path.path[-1].pose.position.x
        self.target_y = self.avoidance_path.path[-1].pose.position.y


       

        self.rviz_avoidance_path.header.stamp = rospy.Time.now()
        
        self.is_local_waypoint_set = True
        self.avoidance_in_progress = True




    # def make_avoidance_path(self, is_big: bool, ccw, mat_v2g: np.ndarray, mat_g2v: np.ndarray, curr_yaw: float):
    #     target_coord = self.setting_target_point(is_big, ccw, mat_g2v, curr_yaw)
        
    #     self.target_x = target_coord[0]
    #     self.target_y = target_coord[1]
    #     target_waypoint_yaw = target_coord[2]

    #     x = np.array([0, self.target_x]) #target_x 두 배 만큼만 뒤로 연장(직선으로)
    #     y = np.array([0, self.target_y])

    #     if np.any(np.diff(x) <= 0):
    #         rospy.logerr("Error: x must be strictly increasing")
    #         return

    #     dydx = np.array([0, target_waypoint_yaw]) #연장했으니 yaw값도 그대로 가져다가 씀 > 0으로 해도 될 것 같기도

    #     f = CubicHermiteSpline(x, y, dydx)
        
    #     resolution = int((np.hypot(self.target_x, self.target_y)) / 0.2) #연장한 것에 맞게 (기존 target_point까지의 거리 + target_x)해서 0.2로 나눔
    #     path_x = np.linspace(x[0], x[-1], resolution)
    #     path_y = f(path_x)

    #     path = np.array([
    #         path_x,
    #         path_y,
    #         np.zeros(shape=(resolution, )),
    #         np.ones(shape=(resolution, )),
    #     ])

    #     df = f.derivative()
    #     path_ori = df(path_x)
        
    #     path = mat_v2g @ path

    #     self.avoidance_path = PathReference()
    #     self.rviz_avoidance_path.poses = []

    #     for x, y, yaw in zip(path[0], path[1], path_ori):
    #         avoidance_path_element = PathReferenceElement()
    #         avoidance_path_element.pose.position.x = x
    #         avoidance_path_element.pose.position.y = y
    #         avoidance_path_element.pose.position.z = 0.

    #         quat = quaternion_from_euler(0., 0., yaw)
    #         avoidance_path_element.pose.orientation.x = quat[0]
    #         avoidance_path_element.pose.orientation.y = quat[1]
    #         avoidance_path_element.pose.orientation.z = quat[2]
    #         avoidance_path_element.pose.orientation.w = quat[3]
            
    #         avoidance_path_element.mode = 11

    #         self.avoidance_path.path.append(avoidance_path_element)

    #         rviz_local_pose = PoseStamped()
    #         rviz_local_pose.header.frame_id = 'map'
    #         rviz_local_pose.header.stamp = rospy.Time.now()
            
    #         rviz_local_pose.pose.position.x = avoidance_path_element.pose.position.x - self.rviz_offset_x
    #         rviz_local_pose.pose.position.y = avoidance_path_element.pose.position.y - self.rviz_offset_y
    #         rviz_local_pose.pose.position.z = 0.
            
    #         rviz_local_pose.pose.orientation.x = quat[0]
    #         rviz_local_pose.pose.orientation.y = quat[1]
    #         rviz_local_pose.pose.orientation.z = quat[2]
    #         rviz_local_pose.pose.orientation.w = quat[3]
            
    #         self.rviz_avoidance_path.poses.append(rviz_local_pose)


    #     # Target 위치 시각화 마커 생성
    #     target_marker = Marker()
    #     target_marker.header.frame_id = 'map'
    #     target_marker.header.stamp = rospy.Time.now()
    #     target_marker.ns = 'target'
    #     target_marker.id = 0
    #     target_marker.type = Marker.SPHERE
    #     target_marker.action = Marker.ADD

    #     target_marker.pose.position.x = self.avoidance_path.path[-1].pose.position.x - self.rviz_offset_x
    #     target_marker.pose.position.y = self.avoidance_path.path[-1].pose.position.y - self.rviz_offset_y
    #     target_marker.pose.position.z = 0.0

    #     target_marker.pose.orientation = self.avoidance_path.path[-1].pose.orientation

    #     target_marker.scale.x = 0.5  # 화살표 길이
    #     target_marker.scale.y = 0.5
    #     target_marker.scale.z = 0.5

    #     target_marker.color.a = 1.0
    #     target_marker.color.r = 1.0
    #     target_marker.color.g = 0.3
    #     target_marker.color.b = 0.0

    #     self.target_marker_pub.publish(target_marker)


    #     self.rviz_avoidance_path.header.stamp = rospy.Time.now()
    #     self.is_local_waypoint_set = True
    #     self.avoidance_in_progress = True

        


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