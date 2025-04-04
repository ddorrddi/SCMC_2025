#! /usr/bin/env python3

# path 경로를 받기 위함
import sys

# 경로 관련 모듈
from nav_msgs.msg import Path
from erp_drive.msg import PathReference
from erp_drive.msg import PathReferenceElement

# Pose 등등 위치 관련 모듈
from geometry_msgs.msg import PoseStamped

# 오일러 <-> 쿼터니언 변환 관련 모듈
import numpy as np
from tf.transformations import quaternion_from_euler

from constant import Lane, const

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
        
        self.rviz_g_path2 = Path()
        self.rviz_g_path2.header.frame_id = 'map'
        self.rviz_g_path2_pub = rospy.Publisher('/rviz_g_path2', Path, queue_size=1)
        
        self.rviz_offset_x = 0.
        self.rviz_offset_y = 0.

        self.is_path_set = False
        
    def set_g_path(self, file_name): 
        # with open(f'{self.pkg_path}/paths/kcity_semi_path_1019_1.txt', 'r') as f:
        
        with open(f'{self.pkg_path}/paths/kcity_semi_path_1019_1.txt', 'r') as f: 
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
         
                if self.rviz_offset_x == 0 and self.rviz_offset_y == 0:
                    self.rviz_offset_x = g_path1_element.pose.position.x
                    self.rviz_offset_y = g_path1_element.pose.position.y

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

        with open(f'{self.pkg_path}/paths/kcity_semi_path_1019_2.txt', 'r') as f:
            for line in f.readlines():
                x, y, yaw, mode = map(float, line.split())
                quat = quaternion_from_euler(0., 0., yaw)
    
                # 2차선
                g_path2_element = PathReferenceElement()
                g_path2_element.pose.position.x = x
                g_path2_element.pose.position.y = y
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

                # 2차선 시각화
                rviz_pose2 = PoseStamped()
                rviz_pose2.header.frame_id = 'map'
                rviz_pose2.header.stamp = rospy.Time.now()

                rviz_pose2.pose.position.x = x - self.rviz_offset_x
                rviz_pose2.pose.position.y = y - self.rviz_offset_y 
                rviz_pose2.pose.position.z = 0.

                rviz_pose2.pose.orientation.x = quat[0]
                rviz_pose2.pose.orientation.y = quat[1]
                rviz_pose2.pose.orientation.z = quat[2]
                rviz_pose2.pose.orientation.w = quat[3]
                
                self.rviz_g_path2.poses.append(rviz_pose2)

        self.g_path[Lane.ONE] = self.g_path1.path
        self.g_path[Lane.TWO] = self.g_path2.path

        self.g_path[Lane.ONE] += self.g_path[Lane.TWO]

        self.is_path_set = True

    def publish_global_path(self, curr_lane: Lane):
        self.g_path_pub.publish(self.g_path[curr_lane])

        self.rviz_g_path1_pub.publish(self.rviz_g_path1)
        self.rviz_g_path2_pub.publish(self.rviz_g_path2)