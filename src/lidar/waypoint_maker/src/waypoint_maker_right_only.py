#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from waypoint_maker.msg import Waypoint
from lidar_object_detection.msg import ObjectInfo
import numpy as np
from scipy.interpolate import CubicSpline
from std_msgs.msg import Int16

LEFT_OFFSET_DISTANCE = 2.0    # 2.0 -> 2.8 (안쪽 파고들기 방지)
MAX_CONE_DISTANCE = 4.0
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
        self.smoothing_alpha = 0.9

        # 안정성 개선을 위한 추가 변수
        self.no_cone_counter = 0  # 콘이 없는 프레임 카운터
        self.no_cone_threshold = 10  # 10프레임(약 0.33초) 연속 미감지 시 완료
        self.min_detection_count = 10
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
            # 라이다 위치 보정: 라이다가 차량 전방 1.25m에 위치
            cone_x = msg.centerX[i] + 1.25
            cone_y = msg.centerY[i]

            if (cone_y < 0) and (cone_x < 5.0):
                all_right_objects.append({'x': cone_x, 'y': cone_y})

                if cone_x > 0:
                    cone = {'x': cone_x, 'y': cone_y}
                    candidate_cones.append(cone)

        # 콘 감지 상태 업데이트
        if len(candidate_cones) > 0:
            self.no_cone_counter = 0  # 콘이 감지되면 카운터 리셋
            if not self.cone_detected_once:
                self.cone_detected_once = True
                rospy.loginfo("[WaypointMaker] U-turn cone detection started")
        else:
            # 콘이 없을 때
            if self.cone_detected_once and not self.done_published:
                self.no_cone_counter += 1
                rospy.loginfo_throttle(1.0, f"[WaypointMaker] No cone detected: {self.no_cone_counter}/{self.no_cone_threshold}")

                # 연속으로 threshold 프레임 동안 콘이 없으면 완료
                if self.no_cone_counter >= self.no_cone_threshold:
                    rospy.loginfo("[WaypointMaker] U-turn completed! Publishing waypoint_done signal")
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

        # 콘 사이사이를 선형으로 채워서 Cubic Spline이 과도하게 휘지 않도록 함
        dense_control_x = []
        dense_control_y = []

        for i in range(len(control_x)):
            dense_control_x.append(control_x[i])
            dense_control_y.append(control_y[i])

            # 다음 콘이 있고, 거리가 2m 이상이면 중간 지점 추가
            if i < len(control_x) - 1:
                dist = np.hypot(control_x[i+1] - control_x[i], control_y[i+1] - control_y[i])
                if dist > 1.0:  # 2m 이상 떨어진 콘 사이에 중간점 추가
                    mid_x = (control_x[i] + control_x[i+1]) / 2.0
                    mid_y = (control_y[i] + control_y[i+1]) / 2.0
                    dense_control_x.append(mid_x)
                    dense_control_y.append(mid_y)

        control_x = dense_control_x
        control_y = dense_control_y
        rospy.loginfo_throttle(2.0, f"[WaypointMaker] Densified control points: {len(control_x)} points")

        control_points = np.array([control_x, control_y]).T
        distances = np.zeros(len(control_points))
        for i in range(1, len(control_points)):
            distances[i] = distances[i-1] + np.hypot(
                control_points[i][0] - control_points[i-1][0],
                control_points[i][1] - control_points[i-1][1]
            )

        try:
            # Cubic Spline 보간 (콘 사이를 미리 채웠으므로 안정적)
            cs_x = CubicSpline(distances, control_x, bc_type='natural')  # natural boundary로 과도한 곡선 억제
            cs_y = CubicSpline(distances, control_y, bc_type='natural')

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
