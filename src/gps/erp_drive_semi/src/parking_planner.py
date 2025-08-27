#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from erp_drive.msg import PathReference, PathReferenceElement
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from std_msgs.msg import Int16
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker

class ParkingPlanner:
    def __init__(self, rviz_offset_x, rviz_offset_y):
        rospy.loginfo("Parking Planner Node Started")

        self.rviz_offset_x = rviz_offset_x
        self.rviz_offset_y = rviz_offset_y

        # Publisher
        self.parking_path_pub = rospy.Publisher('/parking_path', PathReference, queue_size=1)
        self.parking_path_done_pub = rospy.Publisher('/parking_path_done', Int16, queue_size=1)
        self.rviz_parking_path_pub = rospy.Publisher('/rviz_parking_path', Path, queue_size=1)
        self.parking_target_marker_pub = rospy.Publisher('/parking_target_marker', Marker, queue_size=1)
        
        # 후진 경로 관련 Publisher
        self.reverse_path_pub = rospy.Publisher('/reverse_parking_path', PathReference, queue_size=1)
        self.rviz_reverse_path_pub = rospy.Publisher('/rviz_reverse_parking_path', Path, queue_size=1)
        self.reverse_path_done_pub = rospy.Publisher('/reverse_parking_done', Int16, queue_size=1)

        # Subscriber
        self.path_meter_sub = rospy.Subscriber('/parking_path_meter', Path, self.path_meter_callback)
        self.current_pose_sub = rospy.Subscriber('/current_pose', Pose, self.pose_callback)

        # 변수 초기화
        self.is_parking_path_set = False # 사용자가 요청한 플래그
        self.parking_path_global = PathReference() # 생성된 '고정된' 전역 경로
        self.reverse_path_global = PathReference() # 후진용 역순 경로
        self.current_pose = None
        self.final_destination_global = None
        self.reverse_start_global = None # 후진 시작점 (주차 완료 지점)
        self.completion_threshold = 0.3 # 30cm 이내로 들어오면 도착으로 간주
        self.reverse_completion_threshold = 3.0  # 후진은 50cm 이내

        self.parking_mission_complete = False
        self.is_reversing = False # 현재 후진 중인지 플래그

    def path_meter_callback(self, msg):
        # 이미 주차 경로를 따라가는 중이거나, 받은 정보가 유효하지 않거나, 현재 위치를 모르면 무시
        if self.is_parking_path_set or len(msg.poses) < 2 or self.current_pose is None:
            return

        rospy.loginfo("Received new parking path points. Generating fixed global path...")

        # 1. bev_node가 보낸 로컬 좌표계 기준 시작점/끝점을 추출
        start_local = msg.poses[0].pose.position
        end_local = msg.poses[1].pose.position

        # 2. 두 점 사이를 20개의 웨이포인트로 보간
        num_waypoints = 20
        local_waypoints_x = np.linspace(start_local.x, end_local.x, num_waypoints)
        local_waypoints_y = np.linspace(start_local.y, end_local.y, num_waypoints)

        # 3. 경로의 방향(yaw) 계산
        path_dx_local = end_local.x - start_local.x
        path_dy_local = end_local.y - start_local.y
        path_yaw_local = np.arctan2(path_dy_local, path_dx_local)
        quat = quaternion_from_euler(0, 0, path_yaw_local)

        # 4. 현재 차량의 '전역' 위치와 방향을 기준으로 로컬 경로를 '고정된 전역 경로'로 변환
        self.parking_path_global = PathReference() # 경로 초기화
        g_x, g_y = self.current_pose.position.x, self.current_pose.position.y
        orientation_q = self.current_pose.orientation
        _, _, g_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

        for lx, ly in zip(local_waypoints_x, local_waypoints_y):
            # 2D 회전 변환 행렬 적용
            global_x = g_x + lx * np.cos(g_yaw) - ly * np.sin(g_yaw)
            global_y = g_y + lx * np.sin(g_yaw) + ly * np.cos(g_yaw)

            element = PathReferenceElement()
            element.pose.position.x = global_x
            element.pose.position.y = global_y
            
            # 경로의 방향은 차량의 현재 방향에 로컬 경로의 방향을 더한 값
            final_quat = quaternion_from_euler(0, 0, g_yaw + path_yaw_local)
            element.pose.orientation.x, element.pose.orientation.y, element.pose.orientation.z, element.pose.orientation.w = final_quat
            element.mode = 11 # 로컬 경로와 동일한 모드
            self.parking_path_global.path.append(element)
        
        # 5. 최종 목적지의 전역 좌표 저장 (도착 확인용)
        last_point = self.parking_path_global.path[-1].pose.position
        self.final_destination_global = (last_point.x, last_point.y)
        rospy.loginfo(f"Parking path generated. Final destination: ({last_point.x:.2f}, {last_point.y:.2f})")

        # 6. 후진용 역순 경로 생성
        self.generate_reverse_path()
        
        # 7. 플래그 설정: 이제부터 이 경로를 따라 주행 시작
        self.is_parking_path_set = True

    def pose_callback(self, msg):
        self.current_pose = msg
        self.check_completion()

    def check_completion(self):
        # 주차 완료 체크
        if self.is_parking_path_set and self.final_destination_global is not None and self.current_pose is not None:
            dist = np.hypot(
                self.current_pose.position.x - self.final_destination_global[0],
                self.current_pose.position.y - self.final_destination_global[1]
            )
            if dist < self.completion_threshold:
                rospy.loginfo(f"주차 완료! 목표 지점과의 거리: {dist:.2f}m")
                self.parking_path_done_pub.publish(Int16(data=1))
                
                # 후진 시작점 저장 (현재 주차 완료 지점)
                self.reverse_start_global = (self.current_pose.position.x, self.current_pose.position.y)
                rospy.loginfo(f"후진 시작점 저장: ({self.reverse_start_global[0]:.2f}, {self.reverse_start_global[1]:.2f})")
                
                self.is_parking_path_set = False # 상태 리셋
                self.final_destination_global = None
                self.parking_path_global = PathReference()
                self.is_reversing = True # 후진 모드 시작

                self.parking_mission_complete = True
                rospy.loginfo("주차 임무가 영구적으로 완료되었습니다. 후진 감지를 시작합니다.")
        
        # 후진 완료 체크
        if self.is_reversing and self.reverse_start_global is not None and self.current_pose is not None:
            # 후진 시작점(주차 완료 지점)에서 충분히 멀어졌는지 확인
            dist_from_start = np.hypot(
                self.current_pose.position.x - self.reverse_start_global[0],
                self.current_pose.position.y - self.reverse_start_global[1]
            )
            
            # 후진 경로의 마지막 지점(A' 대칭점)에 가까워졌는지 확인
            if len(self.reverse_path_global.path) > 0:
                last_point = self.reverse_path_global.path[-1].pose.position
                dist_to_end = np.hypot(
                    self.current_pose.position.x - last_point.x,
                    self.current_pose.position.y - last_point.y
                )
                
                # 반전 패스 목표점까지의 거리 로그
                rospy.loginfo_throttle(1.0, f"반전 패스 목표점까지의 거리: {dist_to_end:.2f}m")
                
                # 충분히 뒤로 갔다가(2m 이상) 끝점 근처에 도착했으면 완료
                if dist_from_start > 2.0 and dist_to_end < self.reverse_completion_threshold:
                    rospy.loginfo(f"후진 완료! 반전 패스 목표점과의 거리: {dist_to_end:.2f}m")
                    self.reverse_path_done_pub.publish(Int16(data=1))
                    self.is_reversing = False
                    self.reverse_start_global = None
                    rospy.loginfo("후진 완료 신호를 보냈습니다.")

    def publish_path_and_viz(self):
        if not self.is_parking_path_set:
            return
        
        if self.parking_mission_complete:
            rospy.loginfo("주차 임무가 완료되어 경로를 발행하지 않습니다.")
            return

        # 1. 제어기에 전역 좌표계 기준의 주차 경로 발행
        self.parking_path_pub.publish(self.parking_path_global)

        # 2. RViz 시각화
        # 경로 시각화
        rviz_path = Path()
        rviz_path.header.frame_id = 'map'
        rviz_path.header.stamp = rospy.Time.now()
        for element in self.parking_path_global.path:
            pose_stamped = PoseStamped()
            pose_stamped.header = rviz_path.header
            pose_stamped.pose.position.x = element.pose.position.x - self.rviz_offset_x
            pose_stamped.pose.position.y = element.pose.position.y - self.rviz_offset_y
            pose_stamped.pose.orientation = element.pose.orientation
            rviz_path.poses.append(pose_stamped)
        self.rviz_parking_path_pub.publish(rviz_path)

        # 목표 지점 마커 시각화
        if self.final_destination_global:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = rospy.Time.now()
            marker.ns, marker.id, marker.type, marker.action = 'parking_target', 0, Marker.SPHERE, Marker.ADD
            marker.pose.position.x = self.final_destination_global[0] - self.rviz_offset_x
            marker.pose.position.y = self.final_destination_global[1] - self.rviz_offset_y
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = 0.7, 0.7, 0.7
            marker.color.a, marker.color.r, marker.color.g, marker.color.b = 0.8, 0.2, 0.8, 1.0
            self.parking_target_marker_pub.publish(marker)
            
        # 반전 패스 목표점 마커 시각화
        if self.is_reversing and len(self.reverse_path_global.path) > 0:
            reverse_target = self.reverse_path_global.path[-1].pose.position
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = rospy.Time.now()
            marker.ns, marker.id, marker.type, marker.action = 'reverse_target', 0, Marker.SPHERE, Marker.ADD
            marker.pose.position.x = reverse_target.x - self.rviz_offset_x
            marker.pose.position.y = reverse_target.y - self.rviz_offset_y
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = 0.7, 0.7, 0.7
            marker.color.a, marker.color.r, marker.color.g, marker.color.b = 0.8, 1.0, 0.2, 0.2  # 초록색
            self.parking_target_marker_pub.publish(marker)

    def generate_reverse_path(self):
        """주차 경로를 기반으로 후진용 역순 경로 생성 - 원래 시작점에 대해 대칭 연장"""
        if len(self.parking_path_global.path) == 0:
            return
            
        rospy.loginfo("Generating reverse parking path...")
        
        # 1. 원래 주차 경로의 시작점(A)과 끝점(B) 추출
        original_start = self.parking_path_global.path[0]  # A
        original_end = self.parking_path_global.path[-1]   # B
        
        # 2. 주차 경로를 역순으로 복사 (B → A)
        self.reverse_path_global = PathReference()
        reversed_path = list(reversed(self.parking_path_global.path))
        
        # 3. A를 기준으로 B의 대칭점 A' 계산: A' = 2*A - B
        symmetric_point = PathReferenceElement()
        symmetric_point.pose.position.x = 2 * original_start.pose.position.x - original_end.pose.position.x
        symmetric_point.pose.position.y = 2 * original_start.pose.position.y - original_end.pose.position.y
        symmetric_point.pose.position.z = original_start.pose.position.z
        symmetric_point.pose.orientation = original_start.pose.orientation
        symmetric_point.mode = original_start.mode
        
        # 4. A에서 A'까지 추가 경로 생성 (A → A')
        # A의 위치
        A_x = original_start.pose.position.x
        A_y = original_start.pose.position.y
        # A'의 위치  
        Ap_x = symmetric_point.pose.position.x
        Ap_y = symmetric_point.pose.position.y
        
        # A에서 A'까지의 거리 계산
        extension_distance = np.sqrt((Ap_x - A_x)**2 + (Ap_y - A_y)**2)
        num_extension_points = max(int(extension_distance / 0.2), 1)  # 0.2m 간격
        
        # A에서 A'까지 보간된 웨이포인트 생성
        extended_points = []
        for i in range(1, num_extension_points + 1):
            ratio = i / num_extension_points
            extended_point = PathReferenceElement()
            extended_point.pose.position.x = A_x + (Ap_x - A_x) * ratio
            extended_point.pose.position.y = A_y + (Ap_y - A_y) * ratio
            extended_point.pose.position.z = original_start.pose.position.z
            extended_point.pose.orientation = original_start.pose.orientation
            extended_point.mode = original_start.mode
            extended_points.append(extended_point)
        
        # 5. 최종 경로: B → A → A'
        self.reverse_path_global.path = reversed_path + extended_points
            
        rospy.loginfo(f"Generated reverse path with {len(self.reverse_path_global.path)} waypoints")
        
        # 3. 후진 경로 시각화 및 발행
        self.publish_reverse_path()
        
    def publish_reverse_path(self):
        """후진 경로 발행 및 시각화"""
        if len(self.reverse_path_global.path) == 0:
            return
            
        # 1. 제어용 후진 경로 발행
        self.reverse_path_pub.publish(self.reverse_path_global)
        
        # 2. RViz 시각화
        rviz_reverse_path = Path()
        rviz_reverse_path.header.frame_id = 'map'
        rviz_reverse_path.header.stamp = rospy.Time.now()
        
        for element in self.reverse_path_global.path:
            pose_stamped = PoseStamped()
            pose_stamped.header = rviz_reverse_path.header
            pose_stamped.pose.position.x = element.pose.position.x - self.rviz_offset_x
            pose_stamped.pose.position.y = element.pose.position.y - self.rviz_offset_y
            pose_stamped.pose.position.z = element.pose.position.z
            pose_stamped.pose.orientation = element.pose.orientation
            rviz_reverse_path.poses.append(pose_stamped)
        
        self.rviz_reverse_path_pub.publish(rviz_reverse_path)
        rospy.loginfo(f"Published reverse path visualization with {len(rviz_reverse_path.poses)} points")