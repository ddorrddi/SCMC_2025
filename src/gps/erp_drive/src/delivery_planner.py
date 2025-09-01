#! /usr/bin/env python3

# path 경로를 받기 위함
import sys

# 연산 관련 모듈
import math

# 경로 관련 모듈
from nav_msgs.msg import Path
from erp_drive.msg import PathReference
from erp_drive.msg import PathReferenceElement

# 퓨전전용 메시지
from erp_drive.msg import IoUFusionMsg
from geometry_msgs.msg import PointStamped

# Pose 등등 위치 관련 모듈
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
# 라이다 관련 -> 패키지명 확인 후 수정 
from std_msgs.msg import Int16, String, Bool
# 표지판 전용 메시지
from delivery.msg import DetectYolo
# RViz 시각화를 위한 마커 메시지
from visualization_msgs.msg import Marker

# 오일러 <-> 쿼터니언 변환 관련 모듈
import numpy as np
from tf.transformations import euler_from_quaternion


# ros 관련 모듈
import rospy
import rospkg

# 상수값 받아오기
from constant import const

# 큐를 사용하여 최근 데이터의 경향을 파악
from collections import deque
from collections import Counter


class DeliveryPlanner:
    def __init__(self, g_path: PathReference, rviz_offset_x: float, rviz_offset_y: float):
        # 파라미터 (조절해야됨)
        self.threshold = 0.95 # IOU 신뢰도 임계값
        self.min_queue_size = 20 # 목적지 좌표 신뢰도를 위한 큐 사이즈 (10Hz기준 20이면 2초간의 정보) !!메시지 드랍시 보완 로직이 없음!!
        self.end_offset = 40 # 배달미션이 끝났는가 판단하는 최소 waypoint 개수

        # 도착지 시각화
        self.marker = Marker()
        self.marker.header.frame_id = "map"  # Global 좌표계 기준
        self.rviz_offset_x, self.rviz_offset_y = rviz_offset_x, rviz_offset_y
        self.radius_marker = Marker()
        self.radius_marker.header.frame_id = "map"

        # 상태변수
        self.STOP_IDX = 1348 # 배달미션 B지점 시작하는 글로벌패스 인덱스
        self.segment_len = 120 # 배달미션 B구간 1348 ~ 1348+120
        # self.g_path = g_path# 글로벌패스
        self.g_path = [(e.pose.position.x, e.pose.position.y) for e in g_path] # 글로벌패스
        self.curr_idx = None # 현재 글로벌패스 웨이포인트 인덱스
        self.curr_pose = None # 현재 Pose
        self.delivery_target_num = None # 배달 타겟 번호
        self.sign_coords = None # 목적지 좌표 (x,y,z)
        self.delivery_in_progress = False # 배달 진행중인가 플래그
        self.delivery_done = False # 배달 목적지 좌표에 도착했는가 플래그
        self.delivery_done_once = False # delivery_done 한번만 pub하기 위한 flag
        self.delivery_end = False # 배달미션이 끝났는가 플래그
        self.delivery_path_is_pub = False # 배달경로가 한번이라도 publish됐나 플래그
        self.sign_queue = deque(maxlen=70)
        self.result_queue = deque(maxlen=100)
        self.target_stop_xy = None

        # 표지판 / 정지 좌표 튜플
        self.delivery_spots = [
            {"id": 1, "stop_xy": (935681.498, 1916054.600), "sign_xy": (935679.747, 1916052.994)},
            {"id": 2, "stop_xy": (935679.873, 1916049.764), "sign_xy": (935678.274, 1916048.245)},
            {"id": 3, "stop_xy": (935678.302, 1916044.972), "sign_xy": (935676.561, 1916043.554)},
            {"id": 4, "stop_xy": (935676.609, 1916040.558), "sign_xy": (935674.779, 1916039.041)},
            {"id": 5, "sign_xy": (935674.666, 1916035.814), "stop_xy": (935672.758, 1916034.530)},
            {"id": 6, "sign_xy": (935672.421, 1916031.105), "stop_xy": (935670.595, 1916030.131)},
            {"id": 7, "sign_xy": (935670.405, 1916027.326), "stop_xy": (935668.300, 1916025.777)}
        ]

        # 서브스크라이버
        rospy.Subscriber("/current_pose", Pose, self.curr_pose_CB) # 현재 Pose업데이트용
        rospy.Subscriber("/curr_idx", Int16, self.curr_idx_CB) # 현재 idx업데이트용
        rospy.Subscriber('/fusion/matched_center', PointStamped, self.center_point_CB) # 퓨전노드에서 다 처리하고 싹 넘겨받는용
        rospy.Subscriber('/delivery_sign_info', DetectYolo, self.delivery_sign_CB) # 배달목적지 표지판 텍스트

        rospy.Subscriber('/test_delivery', Bool, self.test_CB) # 테스트 서브스크라이버

        # 퍼블리셔
        self.delivery_target_num_pub = rospy.Publisher("/delivery_target", String, queue_size=1)
        self.delivery_done_pub = rospy.Publisher("/delivery_done", Int16, queue_size=1) # 배달 목적지에 도착했는지 플래그
        self.marker_pub = rospy.Publisher("/delivery_target_marker", Marker, queue_size=1) # RViz 시각화를 위한 마커 퍼블리셔


    def test_CB(self, msg:Bool):
        self.sign_coords = (935537.845,
                            1915840.433,
                            0.)

        self.delivery_in_progress = True
        self.get_stop_coords()

    def curr_idx_CB(self, msg: Int16):
        self.curr_idx = msg.data

    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg
        self.delivery_done_confirm()

    def center_point_CB(self, pt_msg: PointStamped):
        # 유효성 체크
        if self.delivery_in_progress:
            return
        if self.curr_idx is None or self.curr_pose is None: # 현재 idx와 Pose가 세팅돼있어야 유효
            return
        if not (self.STOP_IDX <= self.curr_idx <= self.STOP_IDX + self.segment_len): # 배달 B구간일때만 유효
            return

        # 메시지로 넘어온 목적지 좌표(라이다 좌표계)를 글로벌좌표계로 고치기
        ori = [self.curr_pose.orientation.x, self.curr_pose.orientation.y,
               self.curr_pose.orientation.z, self.curr_pose.orientation.w]
        _, _, yaw = euler_from_quaternion(ori)
        x_r, y_r, z_r = (self.curr_pose.position.x,
                         self.curr_pose.position.y,
                         self.curr_pose.position.z)
        lx, ly, lz = pt_msg.point.x, pt_msg.point.y, pt_msg.point.z
        x_g = x_r + (lx * np.cos(yaw) - ly * np.sin(yaw))
        y_g = y_r + (lx * np.sin(yaw) + ly * np.cos(yaw))
        z_g = z_r + lz

        # min_queue_size까지 데이터(목적지 좌표) 샘플 수집 **계속 수집할지, 수집이 완료하면 수집 안할지도 고민**
        if len(self.result_queue) < self.min_queue_size:
            self.result_queue.append((x_g, y_g, z_g))
            return

        # 평균치 내고 튀는값(노이즈) 제거
        arr = np.array(self.result_queue)
        mean_pt = arr.mean(axis=0)
        dists = np.linalg.norm(arr - mean_pt, axis=1)
        inliers = arr[dists <= 0.3]
        final_pt = inliers.mean(axis=0) if inliers.size else mean_pt
        self.result_queue.clear()

        # 표지판 좌표 생성
        self.sign_coords = (final_pt[0], final_pt[1], final_pt[2])
        self.delivery_in_progress = True
        self.get_stop_coords()

    # 가장 가까운 이론상의 표지판 좌표 추출
    def get_stop_coords(self):
        if self.sign_coords is None:
            return None
        if not hasattr(self, "delivery_spots") or not self.delivery_spots:
            return None

        x, y = float(self.sign_coords[0]), float(self.sign_coords[1])

        # sign_xy가 유효한 후보만 대상으로 최근접 탐색
        candidates = [s for s in self.delivery_spots if s.get("sign_xy") is not None]
        if not candidates:
            return None

        # numpy로 빠르게 거리 계산
        arr = np.asarray([s["sign_xy"] for s in candidates], dtype=np.float64)  # shape (N,2)
        dx = arr[:, 0] - x
        dy = arr[:, 1] - y
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))

        # stop_xy 저장
        self.target_stop_xy = candidates[idx]["stop_xy"]
        rospy.loginfo(f"가장 가까운 표지판 좌표: {self.sign_coords}, 정지 좌표: {self.target_stop_xy}, idx: {idx}")

    def delivery_sign_CB(self, msg:DetectYolo):
        print(self.delivery_target_num)
        print(self.sign_queue)
        # 유효성 검사 및 타겟이 정해지지 않았을때 1회만 작동
        # if self.curr_idx is None or self.delivery_done or self.delivery_target_num is not None:
        #     return
        # if (self.STOP_IDX <= self.curr_idx <= self.STOP_IDX + self.segment_len): # 배달 B구간일때는 작동안함
        #     return
        # 큐에 샘플 데이터 수집 이후 충분히 수집하면 target 설정하고 publish
        if msg.class_name != "None":
            self.sign_queue.append(int(msg.class_name))
        if len(self.sign_queue) >= self.min_queue_size:
            self.delivery_target_num = Counter(self.sign_queue).most_common(1)[0][0]
            self.delivery_target_num_pub.publish(String(str(self.delivery_target_num)))
            self.sign_queue.clear()

    # 배달 목적지 좌표에 도착했는가 확인하는 함수
    def delivery_done_confirm(self):
        if self.target_stop_xy is None or self.curr_pose is None:
            self.delivery_done = False
            return
        
        stop_x, stop_y = self.target_stop_xy

        dist_to_target = np.hypot(self.curr_pose.position.x - stop_x,\
                                   self.curr_pose.position.y - stop_y)
        

        rospy.loginfo(f"self.delivery_done_once: {self.delivery_done_once}, dist_to_target: {dist_to_target}")
        if not self.delivery_done_once:
            # rospy.loginfo(f"목적지와의 거리 {dist_to_target}")
            if dist_to_target < const.DELIVERY_SUCCESS_RADIUS and self.delivery_in_progress: # @TODO 실험적으로 찾아야 할 값
                rospy.loginfo(f"delivery_done_confirm() : 목적지와의 거리 {dist_to_target}, 목적지 도착!")
                self.delivery_in_progress = False
                self.delivery_done = True
                self.delivery_done_once = True
        
        self.delivery_done_pub.publish(1 if self.delivery_done else 0)

    def publish_marker(self):
        """
        self.sign_coords에 저장된 좌표를 RViz에 CUBE 형태로 시각화합니다.
        """
        if self.sign_coords is None:
            return

        self.marker.header.stamp = rospy.Time.now()
        self.marker.ns = "delivery_target_marker"
        self.marker.id = 0
        self.marker.type = Marker.CUBE
        self.marker.action = Marker.ADD

        # 마커의 위치 설정
        self.marker.pose.position.x = self.sign_coords[0] - self.rviz_offset_x
        self.marker.pose.position.y = self.sign_coords[1] - self.rviz_offset_y
        self.marker.pose.position.z = self.sign_coords[2]
        
        # 마커의 방향 설정 (회전 없음)
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0

        # 마커의 크기 설정 (0.5m 정육면체)
        self.marker.scale.x = 0.5
        self.marker.scale.y = 0.5
        self.marker.scale.z = 0.5

        # 마커의 색상 설정 (빨간색)
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0  # Alpha(투명도)는 0이 아니어야 보입니다.

        # 마커의 지속시간 (0으로 설정하면 영구적으로 표시)
        self.marker.lifetime = rospy.Duration()
        self.marker_pub.publish(self.marker)
        rospy.loginfo(f"Published delivery target marker at {self.sign_coords}")

        # 2. 도착 반경 마커 (Cylinder)
        self.radius_marker.header.stamp = rospy.Time.now()
        self.radius_marker.ns = "delivery_radius_marker"
        self.radius_marker.id = 1 # 다른 ID를 사용해야 두 마커가 모두 표시됨
        self.radius_marker.type = Marker.CYLINDER
        self.radius_marker.action = Marker.ADD
        self.radius_marker.pose.position.x = self.sign_coords[0] - self.rviz_offset_x
        self.radius_marker.pose.position.y = self.sign_coords[1] - self.rviz_offset_y
        self.radius_marker.pose.position.z = self.sign_coords[2] # 바닥에 표시되도록 z 좌표 동일하게 설정
        self.radius_marker.pose.orientation.w = 1.0
        
        # 지름(scale)은 const.DELIVERY_SUCCESS_RADIUS의 2배
        radius = const.DELIVERY_SUCCESS_RADIUS
        self.radius_marker.scale.x = radius * 2.0
        self.radius_marker.scale.y = radius * 2.0
        self.radius_marker.scale.z = 0.1 # 납작한 원처럼 보이도록 높이는 낮게 설정
        
        # 색상은 반투명한 파란색
        self.radius_marker.color.r = 0.0
        self.radius_marker.color.g = 0.5
        self.radius_marker.color.b = 1.0
        self.radius_marker.color.a = 0.4 # 반투명
        self.radius_marker.lifetime = rospy.Duration()
        self.marker_pub.publish(self.radius_marker)