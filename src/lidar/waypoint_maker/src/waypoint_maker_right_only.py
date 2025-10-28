#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from waypoint_maker.msg import Waypoint
from lidar_object_detection.msg import ObjectInfo
import numpy as np
from scipy.interpolate import CubicSpline
from std_msgs.msg import Int16

LEFT_OFFSET_DISTANCE = 2.0     
MAX_CONE_DISTANCE = 3.3         
INTERPOLATION_STEP = 0.2        

class WaypointMaker:
    def __init__(self):
        self.right_cones = []
        self.waypoint_msg = Waypoint()
        self.done_published = False
        self.cone_detected_once = False
        self.g_curr_idx = 0

        self.prev_waypoint_x = None
        self.prev_waypoint_y = None
        self.smoothing_alpha = 0.7 

        
        rospy.init_node('waypoint_maker', anonymous=True)
        self.cone_sub = rospy.Subscriber("/obstacle_info_semi", ObjectInfo, self.traffic_cone_callback)
        self.idx_sub = rospy.Subscriber("/curr_idx", Int16, self.idx_callback)
        self.waypoint_pub = rospy.Publisher("/waypoint_info", Waypoint, queue_size=1)
        self.cone_visualizer_pub = rospy.Publisher("/cone_marker", MarkerArray, queue_size=1)
        self.waypoint_done_pub = rospy.Publisher("/waypoint_done", Int16, queue_size=1)



    def idx_callback(self, msg):
        self.g_curr_idx = msg.data

    def traffic_cone_callback(self, msg):

        if self.g_curr_idx < 1000 or self.g_curr_idx > 1200:
            return  
        
        # if self.g_curr_idx < 75:
        #     return
        
        self.filter_right_cones(msg)
        self.generate_waypoint_info()
        self.visualize_cones()

    def filter_right_cones(self, msg):
        all_right_objects = []
        candidate_cones = []

        for i in range(msg.objectCounts):
            if (msg.centerY[i] < 0) and (msg.centerX[i] < 3.0):
                all_right_objects.append({'x': msg.centerX[i], 'y': msg.centerY[i]})

                if msg.centerX[i] > 0:
                    cone = {'x': msg.centerX[i], 'y': msg.centerY[i]}
                    candidate_cones.append(cone)

        if self.cone_detected_once and len(candidate_cones) == 0:
            if not self.done_published:
                self.waypoint_done_pub.publish(Int16(data=1))
                self.done_published = True

        candidate_cones.sort(key=lambda c: c['x'])

        filtered_cones = []
        for cone in candidate_cones:
            if not filtered_cones:
                filtered_cones.append(cone)
            else:
                last_cone = filtered_cones[-1]
                distance_to_last = np.hypot(cone['x'] - last_cone['x'], cone['y'] - last_cone['y'])

                if distance_to_last <= MAX_CONE_DISTANCE:
                    filtered_cones.append(cone)

        self.right_cones = filtered_cones

    def generate_waypoint_info(self):
        if not self.right_cones:
            return

        if not self.cone_detected_once:
            self.cone_detected_once = True

        control_x = []
        control_y = []
        for cone in self.right_cones:
            control_x.append(cone['x'])
            control_y.append(cone['y'] + LEFT_OFFSET_DISTANCE)

        if len(control_x) < 3:
            self.waypoint_msg.cnt = len(control_x)
            self.waypoint_msg.x_arr = control_x
            self.waypoint_msg.y_arr = control_y
            self.waypoint_pub.publish(self.waypoint_msg)
            return

        control_points = np.array([control_x, control_y]).T
        distances = np.zeros(len(control_points))
        for i in range(1, len(control_points)):
            distances[i] = distances[i-1] + np.hypot(
                control_points[i][0] - control_points[i-1][0],
                control_points[i][1] - control_points[i-1][1]
            )

        try:
            cs_x = CubicSpline(distances, control_x)
            cs_y = CubicSpline(distances, control_y)

            total_distance = distances[-1]
            num_points = int(total_distance / INTERPOLATION_STEP) + 1
            sample_distances = np.linspace(0, total_distance, num_points)

            interpolated_x = cs_x(sample_distances)
            interpolated_y = cs_y(sample_distances)

            if self.prev_waypoint_x is not None and self.prev_waypoint_y is not None:
                
                prev_len = len(self.prev_waypoint_x)
                curr_len = len(interpolated_x)

                if prev_len == curr_len:
                    
                    interpolated_x = self.smoothing_alpha * interpolated_x + (1 - self.smoothing_alpha) * self.prev_waypoint_x
                    interpolated_y = self.smoothing_alpha * interpolated_y + (1 - self.smoothing_alpha) * self.prev_waypoint_y
                elif prev_len > 5 and curr_len > 5:
                    
                    blend_len = min(prev_len, curr_len)
                    for i in range(blend_len):
                        interpolated_x[i] = self.smoothing_alpha * interpolated_x[i] + (1 - self.smoothing_alpha) * self.prev_waypoint_x[i]
                        interpolated_y[i] = self.smoothing_alpha * interpolated_y[i] + (1 - self.smoothing_alpha) * self.prev_waypoint_y[i]

            self.prev_waypoint_x = interpolated_x.copy()
            self.prev_waypoint_y = interpolated_y.copy()
            
            self.waypoint_msg.cnt = len(interpolated_x)
            self.waypoint_msg.x_arr = interpolated_x.tolist()
            self.waypoint_msg.y_arr = interpolated_y.tolist()

        except Exception as e:
            
            self.waypoint_msg.cnt = len(control_x)
            self.waypoint_msg.x_arr = control_x
            self.waypoint_msg.y_arr = control_y

        
        self.waypoint_pub.publish(self.waypoint_msg)

    def visualize_cones(self):
        
        marker_array = MarkerArray()
        for i, cone in enumerate(self.right_cones):
            marker = Marker()
            marker.header.frame_id = "velodyne"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.8
            marker.color.a = 0.5
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0  
            marker.pose.position.x = cone['x']
            marker.pose.position.y = cone['y']
            marker.pose.position.z = 0.0
            marker_array.markers.append(marker)

        self.cone_visualizer_pub.publish(marker_array)


if __name__ == '__main__':
    try:
        WaypointMaker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
