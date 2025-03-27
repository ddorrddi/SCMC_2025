import math
import rospy
import numpy as np
from std_msgs.msg import Int16
from geometry_msgs.msg import Pose
from lidar_object_detection.msg import ObjectInfo

class ParkingPlanner:
    def __init__(self, total_length=4.835*3, num_rows=13):
        # 주차 공간 관련 변수 초기화
        self.g_curr_idx = None
        self.total_length = total_length  # 전체 주차 공간의 길이
        self.num_rows = num_rows  # 주차 공간의 행 수
        self.row_length = total_length / (num_rows - 1)  # 각 주차 공간의 길이
        self.column_status = ['*'] * num_rows  # 주차 공간 상태 ('*': 비어있음, '#': 차있음)
        self.first_obstacle_utmk = None  # 첫 번째 감지된 장애물의 UTM-K 좌표
        self.last_obstacle_utmk = None  # 마지막으로 감지된 장애물의 UTM-K 좌표
        self.curr_pose = None  # 현재 차량의 위치
        self.stored_obstacles = []  # 저장된 장애물 정보

        self.parking_space_num = Int16()
        self.parking_space_num.data = -1
        self.parking_space_num_pub = rospy.Publisher("/parking_space_num", Int16, queue_size=1)

        # rospy.loginfo(f"주차 공간 초기화: 길이={self.total_length:.2f}m, 행 수={self.num_rows}, 행 길이={self.row_length:.2f}m")

        # ROS 토픽 구독
        rospy.Subscriber("/obstacle_info", ObjectInfo, self.obstacle_info_callback)
        rospy.Subscriber("/current_pose", Pose, self.current_pose_callback)

    def update_g_curr_idx(self, g_curr_idx):
        self.g_curr_idx = g_curr_idx

    def current_pose_callback(self, msg):
        # 현재 차량 위치 업데이트
        self.curr_pose = msg

    def obstacle_info_callback(self, msg):
        # 장애물 정보를 받았을 때 호출되는 콜백 함수
        if self.curr_pose is None:
            return

        if self.g_curr_idx is None:
            return

        # @FMTC
        if self.g_curr_idx < 60: # K City : 3401:
            return

        # 장애물 정보를 numpy 배열로 변환
        obstacles = np.array([
            msg.centerX[:msg.objectCounts],
            msg.centerY[:msg.objectCounts],
            msg.centerZ[:msg.objectCounts]
        ]).T

        # GPS와 라이다 간 거리임
        # obstacles[:, 0] += 1.2

        # 장애물 처리 함수 호출
        self.process_obstacles(obstacles)

    def calculate_obstacle_utmk(self, obstacle):
        # 장애물의 상대 좌표를 UTM-K 좌표로 변환
        if self.curr_pose is None:
            rospy.logwarn("차량 위치 정보가 없습니다.")
            return None

        vehicle_x = self.curr_pose.position.x
        vehicle_y = self.curr_pose.position.y
        yaw_rad = math.radians(self.curr_pose.orientation.z)

        # 좌표 변환 수식
        utmk_x = vehicle_x + obstacle[0] * math.cos(yaw_rad) - obstacle[1] * math.sin(yaw_rad)
        utmk_y = vehicle_y + obstacle[0] * math.sin(yaw_rad) + obstacle[1] * math.cos(yaw_rad)

        return np.array([utmk_x, utmk_y, obstacle[2]])

    def calculate_distance(self, point1, point2):
        # 두 점 사이의 2D 거리 계산
        return np.linalg.norm(np.array(point1[:2]) - np.array(point2[:2]))

    def process_obstacles(self, obstacles):
        # 장애물 처리 메인 함수
        vehicle_position = np.array([self.curr_pose.position.x, self.curr_pose.position.y])
        
        # 모든 장애물을 UTM-K 좌표로 변환
        obstacles_utmk = [self.calculate_obstacle_utmk(obs) for obs in obstacles]
        # 차량으로부터 2미터 이내의 장애물만 필터링
        filtered_obstacles = [
            obs for obs in obstacles_utmk
            if self.calculate_distance(vehicle_position, obs[:2]) <= 1.5
        ]

        if not filtered_obstacles:
            # rospy.logwarn("2미터 이내에 감지된 장애물이 없습니다.")
            return

        # 장애물 거리 계산 및 정렬
        if self.first_obstacle_utmk is None:
            distances = [self.calculate_distance(vehicle_position, obs[:2]) for obs in filtered_obstacles]
        else:
            distances = [self.calculate_distance(self.first_obstacle_utmk[:2], obs[:2]) for obs in filtered_obstacles]

        sorted_indices = np.argsort(distances)
        sorted_obstacles_utmk = np.array(filtered_obstacles)[sorted_indices]
        sorted_distances = np.array(distances)[sorted_indices]

        # 정렬된 장애물 정보로 주차 공간 상태 업데이트
        self.log_first_column_status(sorted_obstacles_utmk, sorted_distances)

    def log_first_column_status(self, obstacles_utmk, distances):
        # 첫 번째 열의 주차 공간 상태 업데이트 및 로깅
        if len(obstacles_utmk) == 0:
            # rospy.loginfo("2미터 이내에 장애물이 감지되지 않았습니다.")
            return

        status_changed = False

        # rospy.loginfo("현재 발견한 장애물 정보 (2미터 이내):")
        # for i, (obstacle_utmk, distance) in enumerate(zip(obstacles_utmk, distances)):
        #     rospy.loginfo(f"  장애물 {i+1}: UTM-K(x={obstacle_utmk[0]:.2f}, y={obstacle_utmk[1]:.2f}, z={obstacle_utmk[2]:.2f}), 거리={distance:.2f}m")

        # rospy.loginfo("장애물 처리:")
        for i, (obstacle_utmk, distance) in enumerate(zip(obstacles_utmk, distances)):
            if self.first_obstacle_utmk is None:
                # 첫 번째 장애물 설정
                index = 0
                self.first_obstacle_utmk = obstacle_utmk
                self.last_obstacle_utmk = obstacle_utmk
                self.column_status[0] = '#'
                self.stored_obstacles.append((0, obstacle_utmk))
                status_changed = True
                # rospy.loginfo(f"첫 번째 장애물 설정: UTM-K(x={obstacle_utmk[0]:.2f}, y={obstacle_utmk[1]:.2f}, z={obstacle_utmk[2]:.2f})")
                continue

            distance_from_last = self.calculate_distance(obstacle_utmk[:2], self.last_obstacle_utmk[:2])
            
            # 너무 가까운 장애물은 무시
            if distance_from_last <= 0.1:
                # rospy.loginfo(f"  장애물 {i+1}: 이전 장애물과의 거리({distance_from_last:.2f}m)가 0.1m 이내여서 무시합니다.")
                continue

            # 장애물의 상대적 위치에 따라 인덱스 결정
            distance_from_first = self.calculate_distance(obstacle_utmk[:2], self.first_obstacle_utmk[:2])
            index = min(round(distance_from_first), self.num_rows - 1)

            # 해당 인덱스가 비어있으면 장애물 정보 저장
            if self.column_status[index] == '*':
                self.column_status[index] = '#'
                self.last_obstacle_utmk = obstacle_utmk
                self.stored_obstacles.append((index, obstacle_utmk))
                status_changed = True
                rospy.loginfo(f"  장애물 {i+1}: 인덱스 {index}에 저장, UTM-K(x={obstacle_utmk[0]:.2f}, y={obstacle_utmk[1]:.2f}, z={obstacle_utmk[2]:.2f})")
                # rospy.loginfo(f"    첫 장애물로부터 거리: {distance_from_first:.2f}m")
            # else:
            #     rospy.loginfo(f"  장애물 {i+1}: 인덱스 {index}가 이미 채워져 있어 무시합니다.")

        # 상태가 변경되었으면 업데이트된 상태 출력
        if status_changed:
            self.print_column_status()
            print(index)

            one = 4
            two = 8
            thr = 12

            if self.parking_space_num.data == -1:
                if index >= two:
                    if self.column_status[one:two + 1].count('*') >= 3:
                        self.parking_space_num.data = 1

                    else:
                        self.parking_space_num.data = 2

                elif index >= one:
                    if self.column_status[:one + 1].count('*') >= 3:
                        self.parking_space_num.data = 0

    def print_column_status(self):
        # 주차 공간 상태를 시각적으로 출력
        rospy.loginfo("갱신된 첫 번째 열 상태:")
        status_str = ''.join(self.column_status)
        visual_status = ["┌───┐" if status == '#' else "┌   ┐" for status in self.column_status]
        middle_status = ["│ X │" if status == '#' else "│   │" for status in self.column_status]
        bottom_status = ["└───┘" if status == '#' else "└   ┘" for status in self.column_status]
        
        rospy.loginfo("    " + " ".join(visual_status))
        rospy.loginfo("    " + " ".join(middle_status))
        rospy.loginfo("    " + " ".join(bottom_status))
        rospy.loginfo(f"    {'   '.join(str(i) for i in range(self.num_rows))}")
        rospy.loginfo(f"Raw status: {status_str}")

        # rospy.loginfo("\n저장된 장애물 정보:")
        # for index, obstacle in self.stored_obstacles:
        #     distance = self.calculate_distance(self.first_obstacle_utmk[:2], obstacle[:2])
        #     rospy.loginfo(f"  인덱스 {index}: (UTM-K: x={obstacle[0]:.2f}, y={obstacle[1]:.2f}), "
        #                   f"첫 장애물로부터 거리={distance:.2f}m")

        # if len(self.stored_obstacles) > 1:
        #     total_detected_length = self.calculate_distance(self.stored_obstacles[0][1][:2], self.stored_obstacles[-1][1][:2])
        #     rospy.loginfo(f"\n감지된 전체 길이: {total_detected_length:.2f}m")
        #     rospy.loginfo(f"예상 전체 길이: {self.total_length:.2f}m")

    def run(self):
        # ROS 노드 실행
        rospy.spin()
