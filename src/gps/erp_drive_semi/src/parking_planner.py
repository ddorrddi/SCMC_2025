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
        self.reverse_path_pub = rospy.Publisher('/reverse_parking_path', PathReference, queue_size=1,latch=True)
        self.rviz_reverse_path_pub = rospy.Publisher('/rviz_reverse_parking_path', Path, queue_size=1)
        self.reverse_path_done_pub = rospy.Publisher('/reverse_parking_done', Int16, queue_size=1)

        # Subscriber
        self.current_pose_sub = rospy.Subscriber('/current_pose', Pose, self.pose_callback)

        # 변수 초기화
        self.is_parking_path_set = False # 사용자가 요청한 플래그
        self.parking_path_global = PathReference() # 생성된 '고정된' 전역 경로
        self.reverse_path_global = PathReference() # 후진용 역순 경로
        self.current_pose = None
        self.final_destination_global = None
        self.reverse_start_global = None # 후진 시작점 (주차 완료 지점)
        self.completion_threshold = 0.3 # 30cm 이내로 들어오면 도착으로 간주
        self.reverse_completion_threshold = 4.0  # 후진은 40cm 이내

        self.parking_mission_complete = False
        self.is_reversing = False # 현재 후진 중인지 플래그

        # 주차 경로 로드


    def load_parking_path(self):
        # 경로 초기화
        self.parking_path_global = PathReference() 

        # ROS 파라미터 서버에서 경로 파일 위치를 읽어옵니다.
        # 파라미터가 설정되지 않았을 경우를 대비해 기본값을 지정합니다.
        path_file = '/home/foscar/SCMC_2025/src/gps/erp_drive_semi/paths/parking_3.txt'

        try:
            with open(path_file, 'r') as f:
                rospy.loginfo(f"Loading parking path from file: {path_file}")
                
                # 파일의 각 줄을 읽어 웨이포인트로 변환합니다.
                for line in f.readlines():
                    # 빈 줄이나 주석('#')은 건너뜁니다.
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # 공백으로 구분된 값을 파싱합니다.
                    try:
                        parts = list(map(float, line.split()))
                        if len(parts) != 4:
                            rospy.logwarn(f"Skipping malformed line in path file: {line}")
                            continue

                        x, y, yaw, mode = parts
                        
                        l_yaw = yaw
                        # 새로운 PathReferenceElement 메시지를 생성합니다.
                        element = PathReferenceElement()
                        
                        # position 필드에 x, y, z 값 할당
                        element.pose.position.x = x
                        element.pose.position.y = y
                        element.pose.position.z = 0.0  # z는 0으로 고정

                        # yaw(라디안)를 쿼터니언으로 변환하여 orientation에 할당
                        quat = quaternion_from_euler(0, 0, yaw)
                        element.pose.orientation.x = quat[0]
                        element.pose.orientation.y = quat[1]
                        element.pose.orientation.z = quat[2]
                        element.pose.orientation.w = quat[3]
                        
                        # 모드 할당 (정수로 변환)
                        element.mode = int(mode)
                        
                        # 생성된 요소를 전역 경로 리스트에 추가합니다.
                        self.parking_path_global.path.append(element)

                        self.publish_path_and_viz()

                    except ValueError as e:
                        rospy.logwarn(f"Error parsing line: {line}. Details: {e}")
                        continue

            rospy.loginfo(f"Successfully loaded {len(self.parking_path_global.path)} waypoints.")

            self.final_destination_global=self.parking_path_global.path[-1]
            # 6. 후진용 역순 경로 생성
            self.generate_reverse_path()
            
            # 7. 플래그 설정: 이제부터 이 경로를 따라 주행 시작
            self.is_parking_path_set = True
        
        except IOError as e:
            rospy.logerr(f"Could not read parking path file: {path_file}. Error: {e}")
            # 파일 로딩에 실패했으므로 주행을 시작하면 안됨
            self.is_parking_path_set = False
    

    def pose_callback(self, msg):
        self.current_pose = msg
        self.check_completion()

    def check_completion(self):
        # 주차 완료 체크
        if self.is_parking_path_set and self.final_destination_global is not None and self.current_pose is not None:
            dist = np.hypot(
                self.current_pose.position.x - self.final_destination_global.pose.position.x,
                self.current_pose.position.y - self.final_destination_global.pose.position.y
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
                    rospy.loginfo(f"후진 시작점에서 충분히 멀어짐: {dist_from_start:.2f}m")
                    rospy.loginfo(f"후진 완료! 반전 패스 목표점과의 거리: {dist_to_end:.2f}m")
                    self.reverse_path_done_pub.publish(Int16(data=1))
                    self.is_reversing = False
                    self.reverse_start_global = None
                    rospy.loginfo("후진 완료 신호를 보냈습니다.")

    def publish_path_and_viz(self):
        if not self.is_parking_path_set:
            return
        
        if self.parking_mission_complete:
            # rospy.loginfo("주차 임무가 완료되어 경로를 발행하지 않습니다.")
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
            marker.pose.position.x = self.final_destination_global.pose.position.x - self.rviz_offset_x
            marker.pose.position.y = self.final_destination_global.pose.position.y - self.rviz_offset_y
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
        
        # 2. 주차 경로를 역순으로 복사 (B → A), 마지막 10% 자르기
        self.reverse_path_global = PathReference()
        reversed_path = list(reversed(self.parking_path_global.path))
        # 마지막 10%를 제거 (A 근처 포인트 제거)
        cut_index = int(len(reversed_path) * 0.6)
        reversed_path = reversed_path[:cut_index]

        # 3. A를 기준으로 B의 대칭점 A' 계산 (거리를 절반으로 축소)
        symmetric_point = PathReferenceElement()
        # A-B 벡터의 절반만 사용해서 A' 생성
        symmetric_point.pose.position.x = original_start.pose.position.x + 0.5 * (original_start.pose.position.x - original_end.pose.position.x)
        symmetric_point.pose.position.y = original_start.pose.position.y + 0.5 * (original_start.pose.position.y - original_end.pose.position.y)
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
        
        # A에서 A'까지 보간된 웨이포인트 생성 (지수함수 곡선으로 부드럽게 아래로 꺾임)
        extended_points = []
        for i in range(1, num_extension_points + 1):
            ratio = i / num_extension_points
            extended_point = PathReferenceElement()
            
            # 기본 직선 보간
            base_x = A_x + (Ap_x - A_x) * ratio
            base_y = A_y + (Ap_y - A_y) * ratio
            
            # A 지점에서 기울기 0으로 시작해서 점점 가파르게 꺾이는 오프셋 (제곱 함수 사용)
            # ratio=0에서 0이고 기울기도 0, ratio=1에서 최대값
            curve_offset = (ratio ** 2.5) * 4.0  # A지점에서 부드럽게 시작해서 점점 가파르게

            extended_point.pose.position.x = base_x
            extended_point.pose.position.y = base_y - curve_offset  # Y축 음의 방향으로 오프셋
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