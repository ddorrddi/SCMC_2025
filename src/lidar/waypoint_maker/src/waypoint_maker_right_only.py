#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from waypoint_maker.msg import Waypoint
from lidar_trafficcone_detection.msg import ObjectInfo
import numpy as np

LEFT_OFFSET_DISTANCE = 2.0      # 좌측 오프셋 거리
MAX_CONE_DISTANCE = 3.3         # 콘 추가 시 거리 기준

class WaypointMaker:
    def __init__(self):
        self.right_cones = []
        self.waypoint_msg = Waypoint()
        
        # ROS 초기화 및 설정
        rospy.init_node('waypoint_maker', anonymous=True)
        self.cone_sub = rospy.Subscriber("/obstacle_info", ObjectInfo, self.traffic_cone_callback)
        self.waypoint_pub = rospy.Publisher("/waypoint_info", Waypoint, queue_size=1)
        self.cone_visualizer_pub = rospy.Publisher("/cone_marker", MarkerArray, queue_size=1)

    def traffic_cone_callback(self, msg):
        self.filter_right_cones(msg)
        self.generate_waypoint_info()
        self.visualize_cones()

    def filter_right_cones(self, msg):
        # 거리 기준을 충족하는 콘만 필터링하여 추가
        filtered_cones = []
        for i in range(msg.objectCounts):
            if (msg.centerY[i] < 0) and (msg.centerX[i] < 2.5):  # Y 좌표가 음수면 우측에 위치
                cone = {'x': msg.centerX[i], 'y': msg.centerY[i]}

                # 리스트가 비어 있으면 첫 번째 콘은 무조건 추가
                if not filtered_cones:
                    filtered_cones.append(cone)
                else:
                    # 마지막 추가된 콘과의 거리 계산
                    last_cone = filtered_cones[-1]
                    distance_to_last = np.hypot(cone['x'] - last_cone['x'], cone['y'] - last_cone['y'])
                    
                    # 거리 기준을 만족할 때만 추가
                    if (distance_to_last <= MAX_CONE_DISTANCE) and (cone['x'] > 0):
                        filtered_cones.append(cone)

        self.right_cones = filtered_cones

    def generate_waypoint_info(self):
        if not self.right_cones:
            return  # 우측 콘이 없으면 종료
        
        # 초기화
        self.waypoint_msg.cnt = len(self.right_cones)
        self.waypoint_msg.x_arr = []
        self.waypoint_msg.y_arr = []

        # 경로 포인트 생성
        for cone in self.right_cones:
            waypoint_x = cone['x']
            waypoint_y = cone['y'] + LEFT_OFFSET_DISTANCE  # 왼쪽으로 이동
            self.waypoint_msg.x_arr.append(waypoint_x)
            self.waypoint_msg.y_arr.append(waypoint_y)

        # 경로 퍼블리싱
        self.waypoint_pub.publish(self.waypoint_msg)

    def visualize_cones(self):
        # Rviz에 표시할 MarkerArray 생성
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
            marker.color.b = 1.0  # Blue for right cones
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
