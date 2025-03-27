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

from static_obstacle_detection.msg import ObstacleInfo

# ros 관련 모듈
import rospy
import rospkg

class GlobalPathPlanner: 
    def __init__(self, pkg_name: str):
        self.pkg_path = rospkg.RosPack().get_path(pkg_name)
        rospy.loginfo(f"pkg_path_is {self.pkg_path}")
        self.g_path = dict()
        self.g_path1 = PathReference()
        self.g_path2 = PathReference()
        self.g_path_pub = rospy.Publisher('/global_path', PathReference, queue_size=1)

        self.rviz_g_path1 = Path()
        self.rviz_g_path1.header.frame_id = 'map'
        self.rviz_g_path1_pub = rospy.Publisher('/rviz_g_path', Path, queue_size=1)
        
        self.rviz_g_path11 = Path()
        self.rviz_g_path11.header.frame_id = 'map'
        self.rviz_g_path11_pub = rospy.Publisher('/rviz_g_path11', Path, queue_size=1)
        
        self.rviz_g_path13 = Path()
        self.rviz_g_path13.header.frame_id = 'map'
        self.rviz_g_path13_pub = rospy.Publisher('/rviz_g_path13', Path, queue_size=1)

        self.rviz_g_path2 = Path()
        self.rviz_g_path2.header.frame_id = 'map'
        self.rviz_g_path2_pub = rospy.Publisher('/rviz_g_path2', Path, queue_size=1)
        
        self.rviz_offset_x = 0.
        self.rviz_offset_y = 0.

        self.is_path_set = False
        
    def set_g_path(self, file_name: str): 
        with open(f'{self.pkg_path}/paths/{file_name}.txt', 'r') as f:
            for line in f.readlines():
                x, y, yaw, mode = map(float, line.split())
                quat = quaternion_from_euler(0., 0., yaw)
                
                # 1차선
                g_path1_element = PathReferenceElement()
                g_path1_element.pose.position.x = x
                g_path1_element.pose.position.y = y
                g_path1_element.pose.position.z = 0.

                g_path1_element.pose.orientation.x = quat[0]
                g_path1_element.pose.orientation.y = quat[1]
                g_path1_element.pose.orientation.z = quat[2]
                g_path1_element.pose.orientation.w = quat[3]

                g_path1_element.mode = int(mode)
                self.g_path1.path.append(g_path1_element)

                # 2차선
                g_path2_element = PathReferenceElement()
                g_path2_element.pose.position.x = x - const.LANE_WIDTH * np.cos(yaw + np.pi / 2)
                g_path2_element.pose.position.y = y - const.LANE_WIDTH * np.sin(yaw + np.pi / 2)
                g_path2_element.pose.position.z = 0.

                g_path2_element.pose.orientation.x = quat[0]
                g_path2_element.pose.orientation.y = quat[1]
                g_path2_element.pose.orientation.z = quat[2]
                g_path2_element.pose.orientation.w = quat[3]

                g_path2_element.mode = int(mode)
                self.g_path2.path.append(g_path2_element)
         
                # rviz로 보기 위해서는 원점부터 경로가 시작되어야 함.
                if self.rviz_offset_x == 0 and self.rviz_offset_y == 0:
                    self.rviz_offset_x = g_path1_element.pose.position.x
                    self.rviz_offset_y = g_path1_element.pose.position.y

                # 1차선 시각화
                rviz_pose = PoseStamped()

                rviz_pose.header.frame_id = 'map'
                rviz_pose.header.stamp = rospy.Time.now()

                rviz_pose.pose.position.x = x + const.LANE_WIDTH_SMALL * np.cos(yaw + np.pi / 2) - self.rviz_offset_x
                rviz_pose.pose.position.y = y + const.LANE_WIDTH_SMALL * np.sin(yaw + np.pi / 2) - self.rviz_offset_y
                rviz_pose.pose.position.z = 0.

                rviz_pose.pose.orientation.x = quat[0]
                rviz_pose.pose.orientation.y = quat[1]
                rviz_pose.pose.orientation.z = quat[2]
                rviz_pose.pose.orientation.w = quat[3]

                self.rviz_g_path11.poses.append(rviz_pose)
                
                rviz_pose = PoseStamped()

                rviz_pose.header.frame_id = 'map'
                rviz_pose.header.stamp = rospy.Time.now()

                rviz_pose.pose.position.x = x - const.LANE_WIDTH_SMALL * np.cos(yaw + np.pi / 2) - self.rviz_offset_x
                rviz_pose.pose.position.y = y - const.LANE_WIDTH_SMALL * np.sin(yaw + np.pi / 2) - self.rviz_offset_y
                rviz_pose.pose.position.z = 0.

                rviz_pose.pose.orientation.x = quat[0]
                rviz_pose.pose.orientation.y = quat[1]
                rviz_pose.pose.orientation.z = quat[2]
                rviz_pose.pose.orientation.w = quat[3]

                self.rviz_g_path13.poses.append(rviz_pose)

                rviz_pose = PoseStamped()

                rviz_pose.header.frame_id = 'map'
                rviz_pose.header.stamp = rospy.Time.now()

                rviz_pose.pose.position.x = x - self.rviz_offset_x
                rviz_pose.pose.position.y = y - self.rviz_offset_y
                rviz_pose.pose.position.z = 0.

                rviz_pose.pose.orientation.x = quat[0]
                rviz_pose.pose.orientation.y = quat[1]
                rviz_pose.pose.orientation.z = quat[2]
                rviz_pose.pose.orientation.w = quat[3]

                self.rviz_g_path1.poses.append(rviz_pose)

                # 2차선 시각화
                rviz_pose2 = PoseStamped()
                rviz_pose2.header.frame_id = 'map'
                rviz_pose2.header.stamp = rospy.Time.now()

                rviz_pose2.pose.position.x = g_path2_element.pose.position.x - self.rviz_offset_x
                rviz_pose2.pose.position.y = g_path2_element.pose.position.y - self.rviz_offset_y 
                rviz_pose2.pose.position.z = 0.

                rviz_pose2.pose.orientation.x = quat[0]
                rviz_pose2.pose.orientation.y = quat[1]
                rviz_pose2.pose.orientation.z = quat[2]
                rviz_pose2.pose.orientation.w = quat[3]
                
                self.rviz_g_path2.poses.append(rviz_pose2)

        self.g_path[Lane.ONE] = self.g_path1.path
        self.g_path[Lane.TWO] = self.g_path2.path

        self.is_path_set = True
        
        # rospy.loginfo(f'GlobalPath : size of path : {len(self.g_path[Lane.ONE])}')

    def publish_global_path(self, curr_lane: Lane):
        self.g_path_pub.publish(self.g_path[curr_lane])

        self.rviz_g_path1_pub.publish(self.rviz_g_path1)
        self.rviz_g_path2_pub.publish(self.rviz_g_path2)
        self.rviz_g_path11_pub.publish(self.rviz_g_path11)
        self.rviz_g_path13_pub.publish(self.rviz_g_path13)