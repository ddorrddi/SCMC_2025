#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from threading import local
import numpy as np
from std_msgs.msg import Float64,Int16,Float32MultiArray
import rospy
import rospkg
import tf
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from waypoint_maker.msg import Waypoint
from erp_drive.msg import PathReference, PathReferenceElement

linspace_num = 200
# extend_length = 5

class drawing_path():
    def __init__(self):
        global linspace_num

        rospy.init_node('linspace')
        pub1 = rospy.Publisher('/waypoint_maker', MarkerArray, queue_size = 1)
        pub2 = rospy.Publisher('/local_waypoint', Waypoint, queue_size = 1)
        rospy.Subscriber('/waypoint_info', Waypoint, self.callback)
        self.x_array = []
        self.y_array = []
        array = [[] for _ in range(linspace_num)]

        rate = rospy.Rate(60)

        while not rospy.is_shutdown():
            # local_path = PathReference()
            local_path = Waypoint()
            if len(self.x_array) > 1 and len(self.y_array) > 1:

                x = self.x_array
                y = self.y_array
                cnt = self.cnt - 1

                diff_x = np.diff(x)
                diff_y = np.diff(y)
                
                dist_diff = np.hypot(diff_x, diff_y)
                interval = dist_diff.sum() / 0.2

                try:
                    # 경고 숨기기
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', np.RankWarning)
                        z = np.polyfit(x, y, 3)
                except:
                    continue
                
                p = np.poly1d(z)

                # print(2*z[0] + z[1])
                 
                # evaluate on new data points
                x_new = np.linspace(min(x), max(x), int(interval))
                y_new = p(x_new)

                array = [[_x, _y] for _x, _y in zip(x_new, y_new)]
                waypoint_marker_arr = MarkerArray()

                # last_x = x_new[-1]
                # last_y = y_new[-1]
                # slope = 2 * z[0] * last_x + z[1]

                for i in range(len(array)):
                    marker = Marker()
                    marker.header.frame_id = "velodyne" # no slash!!!!!
                    marker.id = i
                    marker.type = marker.CUBE
                    marker.action = marker.ADD
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = 1.0
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.pose.position.x = array[i][0]
                    marker.pose.position.y = array[i][1]
                    marker.pose.position.z = 0.2

                    local_path.x_arr.append(array[i][0] +1.3) #차 앞에서 부터
                    local_path.y_arr.append(array[i][1])

                    marker.lifetime = rospy.Duration(1.0)

                    waypoint_marker_arr.markers.append(marker)
                #
                # local_path.tangent = abs(z[0])

                # for i in range(1, int(extend_length / 0.2) + 1):
                #     ext_x = last_x + i * 0.2
                #     ext_y = last_y + slope * (ext_x - last_x)

                #     marker = Marker()
                #     marker.header.frame_id = "velodyne" # no slash!!!!!
                #     marker.id = len(array) + i
                #     marker.type = marker.CUBE
                #     marker.action = marker.ADD
                #     marker.scale.x = 0.1
                #     marker.scale.y = 0.1
                #     marker.scale.z = 0.1
                #     marker.color.a = 1.0
                #     marker.color.r = 1.0
                #     marker.color.g = 0.0
                #     marker.color.b = 0.0
                #     marker.pose.orientation.w = 1.0
                #     marker.pose.position.x = ext_x
                #     marker.pose.position.y = ext_y
                #     marker.pose.position.z = 0.2

                #     local_path.x_arr.append(ext_x +1.3)
                #     local_path.y_arr.append(ext_y)

                #     marker.lifetime = rospy.Duration(1.0)

                #     waypoint_marker_arr.markers.append(marker)
                #

                pub1.publish(waypoint_marker_arr)

                if len(local_path.x_arr) > 0:
                    pub2.publish(local_path)
            rate.sleep()


    def callback(self, msg):
        rospy.loginfo("I get waypoint info")
        
        self.x_array = list(msg.x_arr)[0:msg.cnt]
        self.y_array = list(msg.y_arr)[0:msg.cnt]
        self.cnt = msg.cnt


if __name__ == '__main__':
    try:
        drawing = drawing_path()
    except rospy.ROSInterruptException:
        pass