#! /usr/bin/env python3

# path 경로를 받기 위함
import sys
import os
sys.path.append(os.path.dirname(__file__))

# 경로 관련 모듈
from erp_drive.msg import PathReference

# Pose 등등 위치 관련 모듈
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

from constant import const, Lane

# 각종 Planner 관련 클래스
from global_path_planner import GlobalPathPlanner
from obstacle_avoidance_planner import ObstacleAvoidancePlanner
# from parking_planner import ParkingPlanner
from delivery_planner import DeliveryPlanner # fusion용 추가
from std_msgs.msg import String, Int16

import numpy as np

# ros 모듈
import rospy
from tabulate import tabulate 

class Planner(object):
    def __init__(self, file_name):
        rospy.init_node('planner', argv=file_name)
        self.rate = rospy.Rate(30)
        
        self.g_path_pub = rospy.Publisher('/global_path', PathReference, queue_size=1)
        self.g_path_planner = GlobalPathPlanner('erp_drive')
        self.g_path_planner.set_g_path(file_name)

        self.obstacle_avoidance_planner = ObstacleAvoidancePlanner(self.g_path_planner.g_path, self.g_path_planner.rviz_offset_x, self.g_path_planner.rviz_offset_y)
        self.delivery_planner = DeliveryPlanner(self.g_path_planner.g_path[Lane.ONE], self.g_path_planner.rviz_offset_x, self.g_path_planner.rviz_offset_y)
        # self.parking_planner = ParkingPlanner()

        self.rviz_curr_pose = PoseStamped()
        self.rviz_curr_pose.header.frame_id = 'map'
        self.rviz_curr_pose_pub = rospy.Publisher('/rviz_curr_pose', PoseStamped, queue_size=1)
        
        self.curr_pose = Pose()
        self.curr_pose_sub = rospy.Subscriber('/current_pose', Pose, self.curr_pose_CB)
        
        self.g_curr_idx = 0
        self.curr_idx_sub = rospy.Subscriber("/curr_idx", Int16, self.g_curr_idx_CB)

        self.logging_pub = rospy.Publisher('/logging_planner', String, queue_size=1)
        

        
    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg
        self.obstacle_avoidance_planner.update_curr_pose(self.curr_pose)

        self.rviz_curr_pose.header.stamp = rospy.Time.now()
        self.rviz_curr_pose.pose.position.x = self.curr_pose.position.x - self.g_path_planner.rviz_offset_x
        self.rviz_curr_pose.pose.position.y = self.curr_pose.position.y - self.g_path_planner.rviz_offset_y
        self.rviz_curr_pose.pose.orientation.x = self.curr_pose.orientation.x
        self.rviz_curr_pose.pose.orientation.y = self.curr_pose.orientation.y
        self.rviz_curr_pose.pose.orientation.z = self.curr_pose.orientation.z
        self.rviz_curr_pose.pose.orientation.w = self.curr_pose.orientation.w
        self.rviz_curr_pose_pub.publish(self.rviz_curr_pose)


    
    def g_curr_idx_CB(self, msg: Int16):
        self.g_curr_idx = msg.data
        self.obstacle_avoidance_planner.update_g_curr_idx(self.g_curr_idx)
        # self.parking_planner.update_g_curr_idx(self.g_curr_idx)
    


if __name__ == '__main__':
    planner = Planner(sys.argv[1])
    
    while not rospy.is_shutdown():
        if planner.g_path_planner.is_path_set:
            planner.g_path_planner.publish_global_path(planner.obstacle_avoidance_planner.curr_lane)

        if planner.obstacle_avoidance_planner.is_local_waypoint_set:
             planner.obstacle_avoidance_planner.publish_avoidance_path()
        
        # planner.parking_planner.parking_space_num_pub.publish(planner.parking_planner.parking_space_num)
        planner.delivery_planner.publish_delivery_path()
            
        log = [
            {
                "curr lane": planner.obstacle_avoidance_planner.curr_lane,
                "avoidance" : planner.obstacle_avoidance_planner.avoidance_in_progress,
                "blocked_idx " : planner.obstacle_avoidance_planner.blocked_idx,
                "curr_idx" : planner.obstacle_avoidance_planner.g_curr_idx
            }
        ]

        log = f"\ncurr lane : {planner.obstacle_avoidance_planner.curr_lane}\n"
        log += f"avoidance : {planner.obstacle_avoidance_planner.avoidance_in_progress}\n"
        log += f"blocked_idx : {planner.obstacle_avoidance_planner.blocked_idx}\n"
        log += f"curr_idx : {planner.obstacle_avoidance_planner.g_curr_idx}\n"
        log += f"obst_type : {planner.obstacle_avoidance_planner.obst_type}\n"
        log += f"obst_len_x : {planner.obstacle_avoidance_planner.obst_len_x}\n"
        log += f"obst_len_y : {planner.obstacle_avoidance_planner.obst_len_y}\n"
        log += f"obst_len_z : {planner.obstacle_avoidance_planner.obst_len_z}\n"
        log += f"obst_angle : {np.arctan2(planner.obstacle_avoidance_planner.obst_y, planner.obstacle_avoidance_planner.obst_x) * 180 / np.pi}\n"

        planner.logging_pub.publish(String(log))
        planner.rate.sleep()