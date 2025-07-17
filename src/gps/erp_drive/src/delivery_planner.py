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

# Pose 등등 위치 관련 모듈
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
# 라이다 관련 -> 패키지명 확인 후 수정 
from std_msgs.msg import Int16, String

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
        self.end_offset = 5

        # 로컬패스 설정
        self.pkg_path = rospkg.RosPack().get_path("erp_drive")
        self.delivery_path = PathReference()
        self.rviz_delivery_path = Path()
        self.rviz_delivery_path.header.frame_id = 'map'
        self.rviz_offset_x, self.rviz_offset_y = rviz_offset_x, rviz_offset_y

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
        self.delivery_end = False # 배달미션이 끝났는가 플래그
        self.sign_queue = deque(maxlen=70)
        self.result_queue = deque(maxlen=100)

        # 서브스크라이버
        rospy.Subscriber("/current_pose", Pose, self.curr_pose_CB) # 현재 Pose업데이트용
        rospy.Subscriber("/curr_idx", Int16, self.curr_idx_CB) # 현재 idx업데이트용
        rospy.Subscriber('/fusion_iou', IoUFusionMsg, self.iou_fusion_CB) # 퓨전노드에서 다 처리하고 싹 넘겨받는용 (예상치 10Hz)
        rospy.Subscriber('/delivery_sign', String, self.delivery_sign_CB) # 배달목적지 표지판 텍스트

        rospy.Subscriber('/now_test', Int16, self.test_now_CB) # 로컬패스 생성확인하는 테스트용 구독자

        # 퍼블리셔
        self.delivery_path_pub = rospy.Publisher("/local_path", PathReference, queue_size=1) # 배달경로
        self.rviz_delivery_path_pub = rospy.Publisher('/rviz_delivery_path', Path, queue_size=1) # 시각화용 배달경로
        self.delivery_done_pub = rospy.Publisher("/delivery_done", Int16, queue_size=1) # 배달 목적지에 도착했는지 플래그
        self.delivery_end_pub = rospy.Publisher("/local_path_done", Int16, queue_size=1) # 배달미션이 끝났는지 플래그

    def test_now_CB(self,msg: Int16):
        tester = msg.data
        if tester == 1:
            rospy.loginfo("test_now_CB()호출 : 테스트 시작")
            self.make_delivery_path(isTest=True)
            return
        else:
            return

    def curr_idx_CB(self, msg: Int16):
        self.curr_idx = msg.data

    def curr_pose_CB(self, msg: Pose):
        self.curr_pose = msg

    def iou_fusion_CB(self, msg: IoUFusionMsg): # TODO : 메시지로 넘어온 number가 현재 설정된 target_number와 같은지 체크가 빠져있어요!!!!
        # 유효성 체크
        if self.delivery_in_progress:
            return
        if self.curr_idx is None or self.curr_pose is None: # 현재 idx와 Pose가 세팅돼있어야 유효
            return
        if not (self.STOP_IDX <= self.curr_idx <= self.STOP_IDX + self.segment_len): # 배달 B구간일때만 유효
            return
        if msg.score < self.threshold: # IOU 스코어가 임계치 이상일때만 유효
            return

        # 메시지로 넘어온 목적지 좌표(라이다 좌표계)를 글로벌좌표계로 고치기
        ori = [self.curr_pose.orientation.x, self.curr_pose.orientation.y,
               self.curr_pose.orientation.z, self.curr_pose.orientation.w]
        _, _, yaw = euler_from_quaternion(ori)
        x_r, y_r, z_r = (self.curr_pose.position.x,
                         self.curr_pose.position.y,
                         self.curr_pose.position.z)
        lx, ly, lz = msg.x, msg.y, msg.z
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

        # 최종 목적지 좌표 설정
        self.sign_coords = (final_pt[0], final_pt[1], final_pt[2])

        # 배달 타겟 잡혀있고 배달 진행중이 아니면 로컬패스 생성
        if self.delivery_target_num is not None and not self.delivery_in_progress:
            self.make_delivery_path()

    def delivery_sign_CB(self, msg: String):
        # 유효성 검사 및 타겟이 정해지지 않았을때 1회만 작동
        if self.curr_idx is None or self.delivery_done or self.delivery_target_num is not None:
            return
        if (self.STOP_IDX <= self.curr_idx <= self.STOP_IDX + self.segment_len): # 배달 B구간일때는 작동안함
            return
        # 큐에 샘플 데이터 수집 이후 충분히 수집하면
        self.sign_queue.append(int(msg.data))
        if len(self.sign_queue) >= self.min_queue_size:
            self.delivery_target_num = Counter(self.sign_queue).most_common(1)[0][0]
            self.sign_queue.clear()

    # 배달 목적지 좌표에 도착했는가 확인하는 함수
    def delivery_done_confirm(self):
        if self.sign_coords is None or self.curr_pose is None:
            self.delivery_done = False
            return
        
        dist_to_target = np.hypot(self.curr_pose.position.x - self.sign_coords[0],\
                                   self.curr_pose.position.y - self.sign_coords[1])

        if dist_to_target < const.DELIVERY_SUCCESS_RADIUS and self.delivery_in_progress: # @TODO 실험적으로 찾아야 할 값
            rospy.loginfo("delivery_done_confirm() : 목적지 도착")
            self.delivery_done = True
        
        self.delivery_done_pub.publish(1 if self.delivery_done else 0)

    # 배달미션이 끝났는지 확인하는 함수
    def delivery_end_confirm(self):
        if not self.delivery_in_progress or self.curr_pose is None: # 배달이 진행중이 아니면 체크안함
            return
        else:
            vehicle = (self.curr_pose.position.x, self.curr_pose.position.y)

            local_idx = self.find_closest_index([(e.pose.position.x, e.pose.position.y) for e in self.delivery_path.path], vehicle) # 현재 로컬패스 인덱스 확인
            if local_idx > len(self.delivery_path.path) - self.end_offset: # 로컬패스에서 end_offset만큼 전만큼 보다 앞에 위치해있으면 배달미션 끝.
                self.delivery_end = True
                self.delivery_in_progress = False
                rospy.loginfo("delivery_end_confirm() : 배달미션 끝")
        self.delivery_end_pub.publish(1 if self.delivery_end else 0)

    # 여기부턴 로컬패스 만들기
    # 현재 좌표로부터 가장 가까운 글로벌 웨이포인트 인덱스 리턴
    def find_closest_index(self, path, point):
        px, py = point
        best_idx, best_dist = 0, float('inf')
        for i, (x, y) in enumerate(path):
            d = math.hypot(x - px, y - py)
            if d < best_dist:
                best_dist, best_idx = d, i
        return best_idx

    # 곡선보정을 위한 복귀 글로벌 웨이포인트 인덱스 리턴
    def compute_symmetric_return(self, path, dest, vehicle):
        rospy.loginfo("compute_symmetric_return()호출 : 대칭위치 계산")
        dest_idx    = self.find_closest_index(path, dest)
        vehicle_idx = self.find_closest_index(path, vehicle)
        delta       = dest_idx - vehicle_idx
        sym_idx     = dest_idx + delta
        sym_idx     = max(0, min(sym_idx, len(path)-1))
        return sym_idx, path[sym_idx]

    # 곡선보정해서 좌표들의 리스트 리턴
    def fit_quartic(self, ctrl_pts, resolution=0.3):
        rospy.loginfo("fit_quartic()호출 : 로컬경로 곡선보정")
        pts = np.array(ctrl_pts)
        ds  = np.hypot(np.diff(pts[:,0]), np.diff(pts[:,1]))
        t   = np.concatenate(([0], np.cumsum(ds)))
        coeff_x = np.polyfit(t, pts[:,0], deg=4)
        coeff_y = np.polyfit(t, pts[:,1], deg=4)
        total_length = t[-1]
        n_samples = max(int(total_length / resolution), 2)
        t_new = np.linspace(0, total_length, n_samples)
        x_new = np.polyval(coeff_x, t_new)
        y_new = np.polyval(coeff_y, t_new)
        return list(zip(x_new, y_new))

    # 로컬경로 생성해 리턴
    def generate_local_delivery_path(self, dest, vehicle, n_offset=5, resolution=0.3):
        rospy.loginfo("generate_local_delivery_path()호출 : 로컬경로 제작중")
        # 1. 글로벌 경로 로드
        path = self.g_path
        # 2. 대칭 복귀점 계산
        idx_sym, return_pt = self.compute_symmetric_return(path, dest, vehicle)
        # 3. 주변 WP: 이전/다음 n_offset
        idx_vehicle = self.find_closest_index(path, vehicle)
        prev_wp  = path[max(idx_vehicle - n_offset, 0)]
        next_wp  = path[min(idx_sym + n_offset, len(path)-1)]
        # 4. 컨트롤 포인트 리스트
        ctrl_pts = [prev_wp, vehicle, dest, return_pt, next_wp]
        # 5. 4차 곡선 피팅 후 좌표 반환
        return self.fit_quartic(ctrl_pts, resolution)
    
    # 실제로 사용할 수 있게 메시지 형태로 변환
    def make_delivery_path(self, isTest=False):
        rospy.loginfo("make_delivery_path()호출 : 로컬경로 제작 시작")
        self.delivery_path.path.clear()
        self.rviz_delivery_path.poses.clear()

        if self.delivery_in_progress:
            rospy.loginfo("make_delivery_path() : 현재 배달미션이 이미 진행중")
            return

        # isTest 플래그를 먼저 확인하도록 로직 변경
        if isTest:
            # 테스트 모드일 경우, destination에 테스트용 좌표를 바로 할당


            destination = (935539.3413405563, 1915909.6614581472)
            rospy.loginfo(f"make_delivery_path() : 테스트용 좌표 초기화, destination({destination[0]}, {destination[1]})")
        else:
            # 테스트 모드가 아닐 경우, self.sign_coords가 유효한지 확인
            if self.sign_coords is None:
                rospy.logwarn("Delivery target coordinates (sign_coords) are not set yet.")
                return # 좌표가 없으면 함수 종료
            destination = (self.sign_coords[0], self.sign_coords[1])
            rospy.loginfo(f"make_delivery_path() : 배달목적지 좌표 초기화, destination({destination[0]}, {destination[1]})")

        # vehicle 좌표는 destination 설정 이후에 가져옴
        if self.curr_pose is None:
            rospy.logwarn("Current pose is not available yet.")
            return
        vehicle = (self.curr_pose.position.x, self.curr_pose.position.y)
        
        local_path = self.generate_local_delivery_path(dest=destination, vehicle=vehicle, n_offset=5, resolution=0.3)


        for x, y in local_path:
            delivery_path_element = PathReferenceElement()
            delivery_path_element.pose.position.x = x
            delivery_path_element.pose.position.y = y
            delivery_path_element.pose.position.z = 0.
            delivery_path_element.pose.orientation.x = 0.
            delivery_path_element.pose.orientation.y = 0.
            delivery_path_element.pose.orientation.z = 0.
            delivery_path_element.pose.orientation.w = 1.
            delivery_path_element.mode = 11 
            self.delivery_path.path.append(delivery_path_element)

            rviz_delivery_pose = PoseStamped()
            rviz_delivery_pose.header.frame_id = 'map'
            rviz_delivery_pose.header.stamp = rospy.Time.now()
            rviz_delivery_pose.pose.position.x = delivery_path_element.pose.position.x - self.rviz_offset_x
            rviz_delivery_pose.pose.position.y = delivery_path_element.pose.position.y - self.rviz_offset_y
            rviz_delivery_pose.pose.position.z = 0.
            rviz_delivery_pose.pose.orientation.x = 0.
            rviz_delivery_pose.pose.orientation.y = 0.
            rviz_delivery_pose.pose.orientation.z = 0.
            rviz_delivery_pose.pose.orientation.w = 1.
            
            self.rviz_delivery_path.poses.append(rviz_delivery_pose)

        rospy.loginfo("make_delivery_path() : 로컬경로 제작 완료")
        rospy.loginfo(f'로컬경로 size : {len(self.delivery_path.path)}')

        self.rviz_delivery_path.header.stamp = rospy.Time.now()
        self.delivery_in_progress = True
        self.publish_delivery_path()

    # 로컬경로 퍼블리쉬
    def publish_delivery_path(self):
        isDone = self.delivery_done_confirm()
        isEnd = self.delivery_end_confirm()
        if self.delivery_in_progress:
            rospy.loginfo("publish_delivery_path() : 로컬경로 publish")
            self.rviz_delivery_path_pub.publish(self.rviz_delivery_path)
            self.delivery_path_pub.publish(self.delivery_path)