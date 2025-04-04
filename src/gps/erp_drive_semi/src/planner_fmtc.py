#! /usr/bin/env python3

# path 경로를 받기 위함
import sys
import os
sys.path.append(os.path.dirname(__file__))

# 경로 관련 모듈
from erp_drive.msg import PathReference

# Pose 등등 위치 관련 모듈
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
from constant import const, Lane

# 각종 Planner 관련 클래스
from global_path_planner_fmtc import GlobalPathPlanner
from obstacle_avoidance_planner import ObstacleAvoidancePlanner
from parking_planner import ParkingPlanner
# from delivery_planner import DeliveryPlanner
from waypoint_maker.msg import Waypoint
from std_msgs.msg import String
from erp_drive.msg import PathReference, PathReferenceElement

import numpy as np

# ros 모듈
import rospy
from tabulate import tabulate 

class Planner(object):
    def __init__(self, file_name):
        rospy.init_node('planner', argv=file_name)
        self.rate = rospy.Rate(30)

        self.g_path_planner = GlobalPathPlanner('erp_drive')
        self.g_path_planner.set_g_path(file_name)
        
        self.rviz_curr_pose = PoseStamped()
        self.rviz_curr_pose.header.frame_id = 'map'
        self.rviz_curr_pose_pub = rospy.Publisher('/rviz_curr_pose', PoseStamped, queue_size=1)
        
        self.curr_pose = None
        self.curr_pose_sub = rospy.Subscriber('/current_pose', Pose, self.curr_pose_CB)

        self.u_path_pub = rospy.Publisher('/local_path', PathReference, queue_size=1)
        self.local_waypoint = rospy.Subscriber('/local_waypoint', Waypoint, self.local_waypoint_CB)
        self.rviz_local_path_pub = rospy.Publisher('/rviz_local_path', Path, queue_size=1)

        self.logging_pub = rospy.Publisher('/logging_planner', String, queue_size=1)
    
        self.u_path = PathReference()
    
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

        
    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg
        
        self.rviz_curr_pose.header.stamp = rospy.Time.now()
        self.rviz_curr_pose.pose.position.x = self.curr_pose.position.x - self.g_path_planner.rviz_offset_x
        self.rviz_curr_pose.pose.position.y = self.curr_pose.position.y - self.g_path_planner.rviz_offset_y
        self.rviz_curr_pose.pose.orientation.x = self.curr_pose.orientation.x
        self.rviz_curr_pose.pose.orientation.y = self.curr_pose.orientation.y
        self.rviz_curr_pose.pose.orientation.z = self.curr_pose.orientation.z
        self.rviz_curr_pose.pose.orientation.w = self.curr_pose.orientation.w
        self.rviz_curr_pose_pub.publish(self.rviz_curr_pose)


    def local_waypoint_CB(self, msg: Waypoint):
        if self.curr_pose is None:
            return
        
        x_arr = np.array(msg.x_arr)
        y_arr = np.array(msg.y_arr)
        
        waypoints = np.array([
            x_arr,
            y_arr,
            np.zeros(shape=x_arr.shape),
            np.ones(shape=x_arr.shape)
        ])
        
        mat_v2g, _, _ = self.get_transfomation_matrix()
        path = (mat_v2g @ waypoints)[:2]
        
        rviz_local_path = Path()
        rviz_local_path.header.frame_id = 'map'
        rviz_local_path.header.stamp = rospy.Time.now()
        
        self.u_path.path = []
        for x, y in zip(path[0], path[1]):
            avoidance_path_element = PathReferenceElement()
            avoidance_path_element.pose.position.x = x
            avoidance_path_element.pose.position.y = y
            avoidance_path_element.pose.position.z = 0.

            rviz_local_path_element = PoseStamped()
            rviz_local_path_element.header.frame_id = 'map'
            rviz_local_path_element.header.stamp = rospy.Time.now()
            
            rviz_local_path_element.pose.position.x = x - self.g_path_planner.rviz_offset_x
            rviz_local_path_element.pose.position.y = y - self.g_path_planner.rviz_offset_y
            rviz_local_path_element.pose.position.z = 0.
            
            rviz_local_path.poses.append(rviz_local_path_element)
            
            # quat = quaternion_from_euler(0., 0., yaw)
            # avoidance_path_element.pose.orientation.x = quat[0]
            # avoidance_path_element.pose.orientation.y = quat[1]
            # avoidance_path_element.pose.orientation.z = quat[2]
            # avoidance_path_element.pose.orientation.w = quat[3]
            
            avoidance_path_element.mode = 11

            self.u_path.path.append(avoidance_path_element)

        self.rviz_local_path_pub.publish(rviz_local_path)
        self.u_path_pub.publish(self.u_path)
    
if __name__ == '__main__':
    planner = Planner(sys.argv[1])
    
    while not rospy.is_shutdown():
        if planner.g_path_planner.is_path_set:
            planner.g_path_planner.publish_global_path(Lane.ONE)

        planner.rate.sleep()
