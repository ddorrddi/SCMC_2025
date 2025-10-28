#! /usr/bin/env python3
#-*- encoding: utf-8 -*-
import os.path as ops
import numpy as np
import torch
import cv2
import time
import math
import os
import matplotlib.pylab as plt
import sys
from tqdm import tqdm
import imageio
import tf
# from dataset.dataset_utils import TUSIMPLE
from model2 import *
from utils.evaluation import gray_to_rgb_emb, process_instance_embedding
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, Imu
from race.msg import drive_values
from driving_es import *
from slidewindow_morai import *
from tf.transformations import euler_from_quaternion
from lidar_object_detection.msg import ObjectInfo
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64, Int64, Bool

class Obstacle_Static:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

class Obstacle_Dynamic:
    def __init__(self, x, y, z, width, height, depth):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.depth = depth

    def distance(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def volume(self):
        return self.width * self.height * self.depth

class PID():
  def __init__(self,kp,ki,kd):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.p_error = 0.0
    self.i_error = 0.0
    self.d_error = 0.0

  def pid_control(self, cte):
    self.d_error = cte-self.p_error
    self.p_error = cte
    self.i_error += cte
    return self.kp*self.p_error + self.ki*self.i_error + self.kd*self.d_error

  ### 추가
  def set_gain(self, kp, ki, kd):
      self.kp = kp
      self.ki = ki
      self.kd = kd
      self.p_error = 0.0
      self.i_error = 0.0
      self.d_error = 0.0

class LanenetDetection:
    def __init__(self):
        rospy.init_node('lanenet_detection_node')

        self.ctrl_cmd_pub = rospy.Publisher('control_value', drive_values, queue_size=1)
        self.ctrl_cmd_msg = drive_values()

        self.bridge = CvBridge()
        self.image = np.empty(shape=[0])
        self.slidewindow = SlideWindow()
        self.x_location = 320
        self.last_x_location = 320

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = '/home/foscar/SCMC_2025_Test/src/camera/lanenet/src/lanenet_.model'
        self.LaneNet_model = Lanenet(2, 4)
        self.LaneNet_model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        self.LaneNet_model.to(self.device)
        self.heading = 0.0

        # --- 시퀀스 제어 플래그 (동적부터 시작하도록 True로 변경) ---
        self.is_dynamic = True

        # 동적 장애물 파라미터
        self.dynamic_stop_flag = False
        self.obstacle_list_dynamic = []
        self.z_change_threshold = 0.1
        self.volume_change_threshold = 0.1
        self.last_obstacle_detected_time = None  # 마지막 장애물 감지 시간
        self.stop_hold_duration = 0.5  # 장애물 사라진 후에도 0.5초 정지 유지

        # 정적 장애물 파라미터
        # self.static_state = 'L'  # 'L': Lane, 'A': Avoid, 'R': Return
        # 정적 장애물 파라미터
        # 파라미터 수정
        self.static_state = 'LANE_FOLLOWING'  # 'LANE_FOLLOWING', 'AVOIDING', 'RETURNING'
        self.avoid_direction = None  # 'left' or 'right'
        self.obstacle_list_static = []
        self.avoid_start_time = None
        self.avoid_min_duration = 0.2# 최소 회피 시간 (초)
        self.waypoints = []
        self.waypoints_index = 0
        self.waypoint_reached_threshold = 60  # 픽셀 단위
        self.avoid_stage = 0
        self.force_line = None  # 'left', 'right', or None
        self.pid = PID(0.1, 0.003, 0.01)

        self.stage1_start_time = None
        self.stage2_start_time = None
        self.stage2_end_time = None

        self.global_path_finish = False

        rospy.Subscriber("/global_path_finish", Bool, self.global_path_finish_CB)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        rospy.Subscriber("imu", Imu, self.imuCB)
        rospy.Subscriber("/bounding_box_static", MarkerArray, self.objectCB_static)
        rospy.Subscriber("/bounding_box_dynamic", MarkerArray, self.objectCB_dynamic)

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            pid = PID(0.1, 0.003, 0.010)

            try:
                if self.image.size == 0 or not self.global_path_finish:
                    rate.sleep()
                    continue

                cv2.imshow("lane_image", self.image)

                # 이미지 처리 및 차선 감지 (공통)
                show_img = self.inference(self.image)
                img_frame = show_img.copy()
                height, width, channel = img_frame.shape
                img_roi = img_frame[300:500, 0:]
                img_filtered = color_filter(img_roi)
                height, width, channel = img_filtered.shape

                # Bird's Eye View 변환
                left_margin, top_margin = 195, 44
                src_points = np.float32([[0, 190], [left_margin, top_margin], [width - left_margin, top_margin], [width, 190]])
                # dst_points 간격을 좁혀서 차선이 더 벌어지게 (width//5 ~ width//5*4)
                dst_points = np.float32([[width//5, 190], [width//5, 0], [width//5*4, 0], [width//5*4, 190]])
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                img_warped = cv2.warpPerspective(img_filtered, matrix, [width, height])
                img_warped = cv2.resize(img_warped, dsize=(640, 480))

                # Sliding Window로 차선 중심 찾기
                _, L, _ = cv2.split(cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS))
                _, img_binary = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY)
                img_masked = region_of_interest(img_binary)

                # 정적 장애물 회피 시퀀스에 따라 force_line 설정
                force_line = None
                if not self.is_dynamic:
                    if self.avoid_stage == 1:
                        force_line = 'right'  # Stage 1: 오른쪽 차선만
                    elif self.avoid_stage >= 2:
                        force_line = 'left'   # Stage 2, 3: 왼쪽 차선만

                out_img, self.x_location, _ = self.slidewindow.slidewindow(img_masked, True, force_line)
                if self.x_location is None:
                    self.x_location = self.last_x_location
                else:
                    self.last_x_location = self.x_location

                img_masked_colored = cv2.cvtColor(img_masked, cv2.COLOR_GRAY2BGR)
                img_blended = cv2.addWeighted(out_img, 1, img_masked_colored, 0.6, 0)
                cv2.imshow('lane_detection', img_blended)

                cv2.waitKey(1)

                motor_msg = 10
                angle = pid.pid_control(self.x_location - 320)

                # --- 1. 동적 장애물 회피 시퀀스 ---
                if self.is_dynamic:
                    obstacle_detected_now = False
                    if len(self.obstacle_list_dynamic) > 0:
                        for obstacle in self.obstacle_list_dynamic:
                            # ★★★ 5m 앞에 유효한 장애물이 있으면 즉시 정지 ★★★
                            if (0.0 < obstacle.x < 5.0) and (-1.1 < obstacle.y < 1.1) and self.is_valid_obstacle(obstacle):
                                print('5m 내 동적 장애물 감지: 즉시 정지!')
                                obstacle_detected_now = True
                                self.last_obstacle_detected_time = time.time()
                                self.dynamic_stop_flag = True
                                break

                    # 장애물 감지 또는 최근 감지 후 유지 시간 이내
                    should_stop = False
                    if obstacle_detected_now:
                        should_stop = True
                    elif self.last_obstacle_detected_time is not None:
                        elapsed = time.time() - self.last_obstacle_detected_time
                        if elapsed < self.stop_hold_duration:
                            should_stop = True
                            print(f'[HOLD] 정지 유지 중... ({elapsed:.2f}s / {self.stop_hold_duration}s)')

                    if should_stop:
                        # 정지 명령 발행
                        self.publishCtrlCmd(0, -radians(angle), 0.3)
                        continue # 다음 루프로 넘어가기

                    # 동적 장애물 미션 완료 후 정적 시퀀스로 전환
                    if self.dynamic_stop_flag and len(self.obstacle_list_dynamic) == 0:
                        print("동적 회피 완료!")
                        time.sleep(2)
                        rospy.loginfo("2초 정지!!")
                        self.is_dynamic = False # 시퀀스 전환!
                        self.avoid_stage = 1  # 정적 Stage 1 시작 (오른쪽 붙어서 주행)
                        motor_msg = 15 # 전환 후 빠르게 출발
                    else:
                        motor_msg = 10

                # --- 2. 정적 장애물 회피 시퀀스 (터널: 오른쪽 붙어서 주행 → 정면 장애물 왼쪽 회피) ---
                if not self.is_dynamic:
                    motor_msg = 10

                    # --- 0단계: 오른쪽 차선에 붙어서 주행 (왼쪽 장애물 자동 회피) ---
                    if self.avoid_stage == 1:
                        # 오른쪽 차선 따라가기 (slidewindow가 오른쪽 차선 기준으로 x_location 반환)

                        # 정면 장애물 감지 (두 번째 오른쪽 장애물)
                        obstacle_to_avoid = None
                        for obs in self.obstacle_list_static:
                            if 2.0 < obs.x < 6.0 and -0.7 < obs.y < 0.7:
                                obstacle_to_avoid = obs
                                break

                        if obstacle_to_avoid is not None:
                            print(f"[DEBUG] 정면 장애물 감지 (오른쪽 장애물) at x={obs.x:.2f}, y={obs.y:.2f}")
                            self.avoid_stage = 2  # 왼쪽 회피 시작
                            self.stage1_start_time = time.time()
                            self.pid.set_gain(0.015, 0.003, 0.01)

                    # --- 1단계: 왼쪽으로 회피 ---
                    elif self.avoid_stage == 2:
                        motor_msg = 5
                        angle = 10  # 왼쪽으로 고정 조향각 (음수 = 왼쪽)
                        print(f"[DEBUG] Avoid Stage 2: Moving left with fixed angle={angle}")

                        if not hasattr(self, 'stage1_start_time'):
                            self.stage1_start_time = time.time()

                        # 시간 경과로 Stage 3 전환
                        if (time.time() - self.stage1_start_time) > 1.0:
                            print("Stage 2: 왼쪽 회피 완료 → Stage 3 진입")
                            self.avoid_stage = 3
                            self.stage1_start_time = None

                    # --- 2단계: 왼쪽 차선 따라가기 ---
                    elif self.avoid_stage == 3:
                        motor_msg = 10
                        # slidewindow가 왼쪽 차선 기준으로 x_location 반환
                        print(f"[DEBUG] Avoid Stage 2: Following left lane, x_location={self.x_location}")

                # 최종 제어값 발행
                servo_msg = -radians(angle)
                self.publishCtrlCmd(motor_msg, servo_msg, 0)


            except Exception as e:
                print(e)

            rate.sleep()

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def objectCB_static(self, msg):
        self.obstacle_list_static = []
        for marker in msg.markers:
            obstacle = Obstacle_Static(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
            self.obstacle_list_static.append(obstacle)
        self.obstacle_list_static.sort(key=lambda o: o.distance())

    def objectCB_dynamic(self, msg):
        self.obstacle_list_dynamic = []
        for marker in msg.markers:
            obstacle = Obstacle_Dynamic(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z, marker.scale.x, marker.scale.y, marker.scale.z)
            self.obstacle_list_dynamic.append(obstacle)
        self.obstacle_list_dynamic.sort(key=lambda o: o.distance())

    def global_path_finish_CB(self, msg: Bool):
        self.global_path_finish = msg.data

    def imuCB(self, msg):
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        self.real_heading = yaw * 180.0 / math.pi
        if self.gt_heading is not None:
            self.local_heading = self.real_heading - self.gt_heading
            if self.local_heading > 180: self.local_heading -= 360
            elif self.local_heading < -180: self.local_heading += 360
        else:
            self.local_heading = None

    def is_valid_obstacle(self, new_obstacle):
        # 이 함수는 센서 노이즈를 걸러내는 역할을 합니다.
        # 첫 장애물이거나 비교 대상이 없을 때는 일단 유효하다고 판단합니다.
        if not self.obstacle_list_dynamic:
            return True

        current_volume = new_obstacle.volume()
        current_z = new_obstacle.z

        # 리스트에 있는 기존 장애물들의 평균값 계산
        avg_volume = sum(obs.volume() for obs in self.obstacle_list_dynamic) / len(self.obstacle_list_dynamic)
        avg_z = sum(obs.z for obs in self.obstacle_list_dynamic) / len(self.obstacle_list_dynamic)

        if avg_z == 0 or avg_volume == 0: # 0으로 나누는 오류 방지
            return True

        z_change_rate = abs(current_z - avg_z) / avg_z
        volume_change_rate = abs(current_volume - avg_volume) / avg_volume

        z_within_range = z_change_rate <= self.z_change_threshold
        volume_within_range = volume_change_rate <= self.volume_change_threshold

        return z_within_range and volume_within_range

    def translate_image(self, image, tx, ty):
        rows, cols = image.shape[:2]
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, translation_matrix, (cols, rows))

    def inference(self, gt_img_org):
        org_shape = gt_img_org.shape
        gt_image = cv2.resize(gt_img_org, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
        gt_image = gt_image / 127.5 - 1.0
        gt_image = torch.tensor(gt_image, dtype=torch.float).permute(2, 0, 1)
        gt_image = gt_image.to(self.device)
        binary_final_logits, _ = self.LaneNet_model(gt_image.unsqueeze(0))
        binary_img = torch.argmax(binary_final_logits.cpu(), dim=1).squeeze().numpy().astype(np.uint8)
        binary_img[0:65,:] = 0
        binary_img[binary_img>0] = 255
        frame = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(frame, (org_shape[1], org_shape[0]))

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.throttle = motor_msg
        # 각도를 degree로 변환 후 -30 ~ 30도로 제한
        steering_deg = servo_msg * 180.0 / math.pi 
        steering_deg = max(-30.0, min(30.0, steering_deg))
        self.ctrl_cmd_msg.steering = steering_deg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

if __name__ == "__main__":
    try:
        lanenet_detection_node = LanenetDetection()
    except rospy.ROSInterruptException:
        pass
