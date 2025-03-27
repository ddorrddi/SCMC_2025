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

# 오일러 <-> 쿼터니언 변환 관련 모듈
import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix

# 라이다 관련 -> 패키지명 확인 후 수정 
from std_msgs.msg import Int16, String
from visualization_msgs.msg import Marker, MarkerArray 
from lidar_object_detection.msg import ObjectInfo

# ros 관련 모듈
import rospy
import rospkg

# 상수값 받아오기
from constant import const

# 큐를 사용하여 최근 데이터의 경향을 파악
from collections import deque
from collections import Counter

class DeliveryPlanner:
    def __init__(self, rviz_offset_x: float, rviz_offset_y: float):
        # 배달 시작 지점 (optimal_throttle.h 참조)
        self.STOP_IDX = 1348

        ############################## Path Settings ################################
        self.pkg_path = rospkg.RosPack().get_path("erp_drive")
        # NOTE 1 : delivery line : 표지판 위치 잡기 위한 reference
        self.delivery_line = [[], [], []]
        self.rviz_delivery_line = Path()
        self.rviz_delivery_line.header.frame_id = 'map'

        # NOTE 2 : 실제로 만들어지는 path는 delivery_path
        self.delivery_path = PathReference()
        self.rviz_delivery_path = Path()
        self.rviz_delivery_path.header.frame_id = 'map'
        self.rviz_offset_x, self.rviz_offset_y = rviz_offset_x, rviz_offset_y
        #############################################################################

        self.is_delivery_waypoint_set = False 
        self.delivery_in_progress = False
        self.delivery_done = False

        self.set_path("kcity_delivery_line") 

        ##### Variables ######
        self.delivery_target_num = None #@TODO Planner에서 변경해주기
        self.delivery_target_x = None 
        self.delivery_target_y = None 
        self.delivery_target_yaw = None 
        self.g_obstacle_info = None 
        self.g_curr_idx = None
        self.delivery_start_idx = 0
        self.delivery_end_idx = 999999

        ##### Subscriber #####
        self.curr_pose = None
        self.curr_pose_sub = rospy.Subscriber("/current_pose", Pose, self.curr_pose_CB)
        self.curr_idx_sub = rospy.Subscriber("/curr_idx", Int16, self.curr_idx_CB)
        self.obstacle_info_sub = rospy.Subscriber('/delivery_info', ObjectInfo, self.obstacle_info_CB)
        self.delivery_sign_sub = rospy.Subscriber('/delivery_sign', String, self.delivery_sign_CB)

        ##### Publisher  ######
        self.rviz_delivery_line_pub = rospy.Publisher("/rviz_delivery_line_pub", Path, queue_size= 1)
        self.delivery_path_pub = rospy.Publisher("/delivery_path", PathReference, queue_size=1)
        
        self.rviz_delivery_path_pub = rospy.Publisher('/rviz_delivery_path', Path, queue_size=1)
        self.delivery_done_pub = rospy.Publisher("/delivery_done", Int16, queue_size=1)
        self.obstacle_makers_pub = rospy.Publisher("/obstacle_marker", MarkerArray, queue_size=1)
        
        ####### Flags #########

        # 최근 20개의 표지판 데이터 저장
        self.sign_queue = deque(maxlen=70)
        # 최소 필요 데이터 수
        self.min_queue_size = 50

    def set_path(self, file_name: str):
        with open(f'{self.pkg_path}/paths/{file_name}.txt', 'r') as f:
            for line in f.readlines():
                x, y, yaw, mode = map(float, line.split())
                
                self.delivery_line[0].append(x)
                self.delivery_line[1].append(y)
                self.delivery_line[2].append(yaw)
                
                ### 시각화용 ####
                rviz_pose = PoseStamped()

                rviz_pose.header.frame_id = 'map'
                rviz_pose.header.stamp = rospy.Time.now()

                rviz_pose.pose.position.x = x - self.rviz_offset_x
                rviz_pose.pose.position.y = y - self.rviz_offset_y
                rviz_pose.pose.position.z = 0.

                quat = quaternion_from_euler(0., 0., yaw)
                rviz_pose.pose.orientation.x = quat[0]
                rviz_pose.pose.orientation.y = quat[1]
                rviz_pose.pose.orientation.z = quat[2]
                rviz_pose.pose.orientation.w = quat[3]

                self.rviz_delivery_line.poses.append(rviz_pose)

            self.rviz_delivery_line.header.stamp = rospy.Time.now()

        self.delivery_line = np.array(self.delivery_line)
        self.is_delivery_waypoint_set = True

    def curr_idx_CB(self, idx : Int16) :
        self.g_curr_idx = idx.data

    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg

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

    # 근데 이럴거면 그냥 message filtering 하는게 나을 거 같은데
    def obstacle_info_CB(self, msg : ObjectInfo):
        if self.g_curr_idx is None or self.curr_pose is None:
            return 
        
        # 혹시나 혹시나 해서 남김 -> 원래 1450임
        if self.STOP_IDX > self.g_curr_idx or self.g_curr_idx > self.STOP_IDX + 120:
            return 

        self.objectCounts = msg.objectCounts
    
        # msg에 내용물이 없는 경우
        if msg.objectCounts == 0:
            return
        
        ### lidar 좌표계 상의 정보를 global 좌표계 상 정보로 변경 ####
        self.rel_obstacle_info = np.array([msg.centerX, msg.centerY, msg.centerZ, msg.lengthX, msg.lengthY, msg.lengthZ])
        self.rel_obstacle_info = self.rel_obstacle_info[:, :msg.objectCounts]
        self.rel_obstacle_info[0] += 1.2
        
        # GPS 수신기의 위치에 따라 값이 변동됨.
        # ERP 기준으로는 ~~ 약 0.2???? @@@ 내일 하기 전에 꼭 변경하기
        # self.rel_obstacle_info[1] -= 0.2

        angle_filter = np.abs(np.arctan2(self.rel_obstacle_info[1], self.rel_obstacle_info[0])) < np.pi / 6
        self.rel_obstacle_info = self.rel_obstacle_info[:, angle_filter]
        
        if len(self.rel_obstacle_info[0]) == 0:
            return
       
        #가장 가까운 순으로 정렬 거리 계산 및 정렬
        distances = np.hypot(self.rel_obstacle_info[0], self.rel_obstacle_info[1])
        sorted_indices = np.argsort(distances)
        self.rel_obstacle_info = self.rel_obstacle_info[:, sorted_indices]

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
        
    # 여기서 distance_threshold가 무슨 말인지 모르겠음
    def find_delivery_target_coord(self, obstacles: np.ndarray):
        dists = np.hypot(obstacles[0] - self.delivery_line[0], obstacles[1] - self.delivery_line[1])
        candi_blocked_idx = np.argmin(dists)
        distance_threshold = 1.0
        
        if dists[candi_blocked_idx] <= distance_threshold:
            self.blocked_idx = np.min(candi_blocked_idx - 5, 0)
            self.delivery_target_x = self.delivery_line[0][self.blocked_idx]
            self.delivery_target_y = self.delivery_line[1][self.blocked_idx]
            self.delivery_target_yaw = self.delivery_line[2][self.blocked_idx]
            return True
        
        return False
    
    def delivery_done_confirm(self):
        if self.delivery_target_x is None and self.delivery_target_y is None:
            return False
        
        dist_to_target = np.hypot(self.curr_pose.position.x - self.delivery_target_x,\
                                   self.curr_pose.position.y - self.delivery_target_x)

        return dist_to_target < const.DELIVERY_SUCCESS_RADIUS # @TODO 실험적으로 찾아야 할 값

    def delivery_sign_CB(self, msg: String):
        if self.g_curr_idx is None or self.curr_pose is None or self.g_obstacle_info is None:
            return

        if self.delivery_target_num is None:
            self.delivery_target_num = int(msg.data)

        # 인덱스 안에 들어오지 않았으면 그냥 리턴
        if self.STOP_IDX > self.g_curr_idx or self.g_curr_idx > self.STOP_IDX +  120:
            return
        
        # 배달 완료 했으면 그냥 리턴 
        if self.delivery_done:
            return
        
        # 이미 delivery path 생성이 된 경우는 더 이상 밑의 내용을 수행하지 말아야함 
        if self.delivery_in_progress:
            # rospy.loginfo('delivery_info_CB: already delivery')
            if self.delivery_done_confirm():
                self.delivery_done = True 
            return
            
        # 카메라에서 받아온 배달 표지판 값 중 최소값으로 내가 어느 영역에 있는지 인식 
        sign_num = int(msg.data)
        self.sign_queue.append(sign_num)
        
        # 충분한 데이터가 쌓였을 때만 처리
        if len(self.sign_queue) >= self.min_queue_size:
            most_common_sign = Counter(self.sign_queue).most_common(1)[0][0]

            if most_common_sign == self.delivery_target_num:
                if len(self.g_obstacle_info) > 0: # 라이다 상에 장애물이 들어오면 
                # 그 장애물이 배달 line에 겹치는지도 확인해야함 > 이상한 장애물 잡았을 수도 있으니
                    if self.find_delivery_target_coord(self.g_obstacle_info): 
                        self.make_delivery_path()

    def make_delivery_path(self):
        mat_v2g, mat_g2v, curr_yaw = self.get_transfomation_matrix()
        rel_x, rel_y, _, _ = mat_g2v @ np.array([self.delivery_target_x, self.delivery_target_y, 0., 1.]).T
        
        yaw = self.delivery_target_yaw - curr_yaw 

        x = np.array([0, rel_x])
        y = np.array([0, rel_y])

        if np.any(np.diff(x)<=0):
            rospy.logerr("Error : x must be strictly increasing")
            return
        
        dydx = np.array([0, yaw])
        f = CubicHermiteSpline(x, y, dydx)
        resolution = int(np.hypot(rel_x, rel_y) / 0.2)
        path_x = np.linspace(x[0], x[-1], resolution)
        path_y = f(path_x)

        path = np.array([
            path_x,
            path_y,
            np.zeros(shape=(resolution, )),
            np.ones(shape=(resolution,)),
        ])
        df = f.derivative()
        path_ori = df(path_x)

        path = mat_v2g @ path

        self.delivery_path = PathReference()
        self.rviz_delivery_path.poses = []
       
        for x, y, yaw in zip(path[0], path[1], path_ori):
            delivery_path_element = PathReferenceElement()
            delivery_path_element.pose.position.x = x
            delivery_path_element.pose.position.y = y
            delivery_path_element.pose.position.z = 0.

            quat = quaternion_from_euler(0., 0., yaw)
            
            delivery_path_element.pose.orientation.x = quat[0]
            delivery_path_element.pose.orientation.y = quat[1]
            delivery_path_element.pose.orientation.z = quat[2]
            delivery_path_element.pose.orientation.w = quat[3]

            delivery_path_element.mode = 11 
            
            self.delivery_path.path.append(delivery_path_element)

            rviz_delivery_pose = PoseStamped()
            rviz_delivery_pose.header.frame_id = 'map'
            rviz_delivery_pose.header.stamp = rospy.Time.now()
            
            rviz_delivery_pose.pose.position.x = delivery_path_element.pose.position.x - self.rviz_offset_x
            rviz_delivery_pose.pose.position.y = delivery_path_element.pose.position.y - self.rviz_offset_y
            rviz_delivery_pose.pose.position.z = 0.

            rviz_delivery_pose.pose.orientation.x = quat[0]
            rviz_delivery_pose.pose.orientation.y = quat[1]
            rviz_delivery_pose.pose.orientation.z = quat[2]
            rviz_delivery_pose.pose.orientation.w = quat[3]
            
            self.rviz_delivery_path.poses.append(rviz_delivery_pose)

        rospy.loginfo(f'path_size {len(self.delivery_path.path)}')

        self.rviz_delivery_path.header.stamp = rospy.Time.now()
        self.delivery_in_progress = True

    def publish_delivery_path(self):
        if self.is_delivery_waypoint_set:
            self.rviz_delivery_line_pub.publish(self.rviz_delivery_line)

        if self.delivery_in_progress:
            self.rviz_delivery_path_pub.publish(self.rviz_delivery_path)
            self.delivery_path_pub.publish(self.delivery_path)
