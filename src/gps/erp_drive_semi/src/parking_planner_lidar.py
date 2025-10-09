#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from erp_drive.msg import PathReference, PathReferenceElement
from geometry_msgs.msg import Pose, PoseStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_matrix
from std_msgs.msg import Int16
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from lidar_object_detection.msg import ObjectInfo

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
        self.reverse_path_pub = rospy.Publisher('/reverse_parking_path', PathReference, queue_size=1, latch=True)
        self.rviz_reverse_path_pub = rospy.Publisher('/rviz_reverse_parking_path', Path, queue_size=1)
        self.reverse_path_done_pub = rospy.Publisher('/reverse_parking_done', Int16, queue_size=1)

        # 후보 경로들 시각화
        self.candidate_paths_pub = rospy.Publisher('/candidate_parking_paths', MarkerArray, queue_size=1)

        # Subscriber
        self.current_pose_sub = rospy.Subscriber('/current_pose', Pose, self.pose_callback)
        self.obstacle_info_sub = rospy.Subscriber('/obstacle_info_semi', ObjectInfo, self.obstacle_info_CB)

        # 변수 초기화
        self.is_parking_path_set = False  # 사용자가 요청한 플래그
        self.parking_path_global = PathReference()  # 생성된 '고정된' 전역 경로
        self.reverse_path_global = PathReference()  # 후진용 역순 경로
        self.current_pose = None
        self.final_destination_global = None
        self.reverse_start_global = None  # 후진 시작점 (주차 완료 지점)
        self.completion_threshold = 0.3  # 30cm 이내로 들어오면 도착으로 간주
        self.reverse_completion_threshold = 4.0  # 후진은 40cm 이내

        self.parking_mission_complete = False
        self.is_reversing = False  # 현재 후진 중인지 플래그

        # 다중 경로 관리
        self.all_parking_paths = {}  # {1: PathReference, 2: PathReference, ...}
        self.selected_path_id = None
        self.path_selection_done = False
        self.blocked_paths = set()  # 한번 막힌 경로 ID 저장 (블랙리스트)

        # 장애물 정보
        self.g_obstacle_info = None  # 글로벌 좌표계 장애물 정보
        self.obstacle_collision_threshold = 0.3  # 0.5m 이내면 충돌로 간주

        # 경로 로드

    def load_parking_path(self):
        """parking_1.txt ~ parking_6.txt 모든 경로 로드"""
        rospy.loginfo("Loading all parking paths...")

        base_path = '/home/foscar/SCMC_2025/src/gps/erp_drive_semi/paths/'

        for i in range(1, 7):
            path_file = f'{base_path}parking_{i}.txt'
            try:
                parking_path = PathReference()

                with open(path_file, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        try:
                            parts = list(map(float, line.split()))
                            if len(parts) != 4:
                                continue

                            x, y, yaw, mode = parts

                            element = PathReferenceElement()
                            element.pose.position.x = x
                            element.pose.position.y = y
                            element.pose.position.z = 0.0

                            quat = quaternion_from_euler(0, 0, yaw)
                            element.pose.orientation.x = quat[0]
                            element.pose.orientation.y = quat[1]
                            element.pose.orientation.z = quat[2]
                            element.pose.orientation.w = quat[3]

                            element.mode = int(mode)
                            parking_path.path.append(element)

                        except ValueError as e:
                            rospy.logwarn(f"Error parsing line in {path_file}: {line}")
                            continue

                self.all_parking_paths[i] = parking_path
                rospy.loginfo(f"Loaded parking_{i}.txt with {len(parking_path.path)} waypoints")

            except IOError as e:
                rospy.logerr(f"Could not read {path_file}: {e}")

        rospy.loginfo(f"Total {len(self.all_parking_paths)} parking paths loaded")

        # 후보 경로들 시각화
        self.visualize_candidate_paths()

    def get_transformation_matrix(self):
        if self.current_pose is None:
            return None, None, None

        orientation_list = [
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        ]

        transition = np.array([
            [1., 0., 0., self.current_pose.position.x],
            [0., 1., 0., self.current_pose.position.y],
            [0., 0., 1., self.current_pose.position.z],
            [0., 0., 0., 1.]
        ])

        inverse_transition = np.array([
            [1., 0., 0., -self.current_pose.position.x],
            [0., 1., 0., -self.current_pose.position.y],
            [0., 0., 1., -self.current_pose.position.z],
            [0., 0., 0., 1.]
        ])

        rotation = quaternion_matrix(orientation_list)

        mat_v2g = transition @ rotation
        mat_g2v = rotation.T @ inverse_transition

        _, _, curr_yaw = euler_from_quaternion(orientation_list)

        return mat_v2g, mat_g2v, curr_yaw

    def obstacle_info_CB(self, msg: ObjectInfo):
        if msg.objectCounts == 0:
            return

        if self.current_pose is None:
            return

        rel_obstacle_info = np.array([
            msg.centerX, msg.centerY, msg.centerZ,
            msg.lengthX, msg.lengthY, msg.lengthZ
        ])
        rel_obstacle_info = rel_obstacle_info[:, :msg.objectCounts]


        ############################## 라이다 위치 보정 ##############################
        # 라이다가 차량 전방 0.8m
        rel_obstacle_info[0] +=1.2

        angle_filter = np.abs(np.arctan2(rel_obstacle_info[1], rel_obstacle_info[0])) < np.pi / 9
        rel_obstacle_info = rel_obstacle_info[:, angle_filter]

        if len(rel_obstacle_info[0]) == 0:
            return

        mat_v2g, _, _ = self.get_transformation_matrix()
        if mat_v2g is None:
            return

        self.g_obstacle_info = np.copy(rel_obstacle_info)

        info = np.array([
            rel_obstacle_info[0],
            rel_obstacle_info[1],
            np.zeros(len(rel_obstacle_info[0])),
            np.ones(len(rel_obstacle_info[0]))
        ])

        self.g_obstacle_info[:2] = (mat_v2g @ info)[:2]

        if not self.path_selection_done:
            self.select_safe_parking_path()
        elif self.path_selection_done and not self.parking_mission_complete:
            if self.is_current_path_blocked():
                self.select_safe_parking_path()

    def is_path_blocked(self, parking_path: PathReference):
        if self.g_obstacle_info is None or len(self.g_obstacle_info[0]) == 0:
            return False

        if len(parking_path.path) == 0:
            return True

        for waypoint in parking_path.path:
            wp_x = waypoint.pose.position.x
            wp_y = waypoint.pose.position.y

            for i in range(len(self.g_obstacle_info[0])):
                obst_x = self.g_obstacle_info[0][i]
                obst_y = self.g_obstacle_info[1][i]
                obst_radius = self.g_obstacle_info[4][i] / 2  # lengthY / 2

                dist = np.hypot(wp_x - obst_x, wp_y - obst_y)

                if dist < (obst_radius + self.obstacle_collision_threshold):
                    return True

        return False

    def is_current_path_blocked(self):
        if self.selected_path_id is None:
            return False

        return self.is_path_blocked(self.parking_path_global)

    def select_safe_parking_path(self):
        if len(self.all_parking_paths) == 0:
            return

        safe_paths = []

        for path_id, parking_path in self.all_parking_paths.items():
            if path_id in self.blocked_paths:
                continue

            if not self.is_path_blocked(parking_path):
                safe_paths.append(path_id)
            else:
                self.blocked_paths.add(path_id)

        if len(safe_paths) == 0:
            return

        selected_id = safe_paths[0]

        if self.selected_path_id == selected_id and self.path_selection_done:
            return

        self.selected_path_id = selected_id
        self.parking_path_global = self.all_parking_paths[selected_id]
        self.final_destination_global = self.parking_path_global.path[-1]

        self.generate_reverse_path()

        self.is_parking_path_set = True
        self.path_selection_done = True

    def pose_callback(self, msg):
        self.current_pose = msg
        self.check_completion()

    def check_completion(self):
        if self.is_parking_path_set and self.final_destination_global is not None and self.current_pose is not None:
            dist = np.hypot(
                self.current_pose.position.x - self.final_destination_global.pose.position.x,
                self.current_pose.position.y - self.final_destination_global.pose.position.y
            )
            if dist < self.completion_threshold:
                rospy.loginfo(f"주차 목표 지점과의 거리: {dist:.2f}m")
                self.parking_path_done_pub.publish(Int16(data=1))

                self.reverse_start_global = (self.current_pose.position.x, self.current_pose.position.y)

                self.is_parking_path_set = False
                self.final_destination_global = None
                self.parking_path_global = PathReference()
                self.is_reversing = True
                self.parking_mission_complete = True
                rospy.loginfo("주차 임무 완료. 후진 감지 시작")

        # 후진 완료 체크
        if self.is_reversing and self.reverse_start_global is not None and self.current_pose is not None:
            dist_from_start = np.hypot(
                self.current_pose.position.x - self.reverse_start_global[0],
                self.current_pose.position.y - self.reverse_start_global[1]
            )

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
            return

        self.parking_path_pub.publish(self.parking_path_global)

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

    def visualize_candidate_paths(self):
        """후보 경로들을 RViz에 시각화"""
        marker_array = MarkerArray()

        colors = [
            (1.0, 0.0, 0.0),  # 빨강
            (0.0, 1.0, 0.0),  # 초록
            (0.0, 0.0, 1.0),  # 파랑
            (1.0, 1.0, 0.0),  # 노랑
            (1.0, 0.0, 1.0),  # 마젠타
            (0.0, 1.0, 1.0),  # 시안
        ]

        for path_id, parking_path in self.all_parking_paths.items():
            for idx, element in enumerate(parking_path.path):
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = rospy.Time.now()
                marker.ns = f'candidate_path_{path_id}'
                marker.id = idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                marker.pose.position.x = element.pose.position.x - self.rviz_offset_x
                marker.pose.position.y = element.pose.position.y - self.rviz_offset_y
                marker.pose.position.z = 0.0
                marker.pose.orientation.w = 1.0

                marker.scale.x, marker.scale.y, marker.scale.z = 0.1, 0.1, 0.1

                color = colors[(path_id - 1) % len(colors)]
                marker.color.r, marker.color.g, marker.color.b = color
                marker.color.a = 0.3

                marker_array.markers.append(marker)

        self.candidate_paths_pub.publish(marker_array)

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
            curve_offset = (ratio ** 1.5) * 4.0  # A지점에서 부드럽게 시작해서 점점 가파르게

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
