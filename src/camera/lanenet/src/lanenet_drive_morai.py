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
# from race.msg import drive_values
from driving_es import *
from slidewindow_morai import *
from tf.transformations import euler_from_quaternion
from lidar_object_detection.msg import ObjectInfo
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64, Int64, Bool

from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage

# drive_values_pub= rospy.Publisher('control_value', drive_values, queue_size = 1)

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

class LanenetDetection:
    def __init__(self):
        rospy.init_node('lanenet_detection_node')

        self.ctrl_cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 2

        self.bridge = CvBridge()
        self.image = np.empty(shape=[0])
        self.slidewindow = SlideWindow()
        self.x_location = 320
        self.last_x_location = 320

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = '/home/foscar/iscc_2024/src/camera/lanenet/src/lanenet_.model'
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
        
        # 정적 장애물 파라미터
        # self.static_state = 'L'  # 'L': Lane, 'A': Avoid, 'R': Return
        # 정적 장애물 파라미터
        self.is_passing_static_zone = False  # [추가] 회피 구간 플래그
        self.real_heading = None
        self.gt_heading = None
        self.avoid_heading = None
        self.return_heading = None
        self.local_heading = None

        self.global_path_finish = False

        rospy.Subscriber("/global_path_finish", Bool, self.global_path_finish_CB)
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback)
        rospy.Subscriber("imu", Imu, self.imuCB)
        rospy.Subscriber("/bounding_box_static", MarkerArray, self.objectCB_static)
        rospy.Subscriber("/bounding_box_dynamic", MarkerArray, self.objectCB_dynamic)

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            pid = PID(0.015, 0.003, 0.010)

            try:
                if self.image.size == 0 or not self.global_path_finish:
                    rate.sleep()
                    continue

                cv2.imshow("lane_image", self.image)

                # 이미지 처리 및 차선 감지 (공통)
                show_img = self.inference(self.image)
                img_frame = show_img.copy()
                height, width, channel = img_frame.shape
                img_roi = img_frame[280:470, 0:]
                img_filtered = color_filter(img_roi)
                height, width, channel = img_filtered.shape

                # Bird's Eye View 변환
                left_margin, top_margin = 195, 44
                src_points = np.float32([[0, 190], [left_margin, top_margin], [width - left_margin, top_margin], [width, 190]])
                dst_points = np.float32([[width//4, 190], [width//4, 0], [width//4*3, 0], [width//4*3, 190]])
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                img_warped = cv2.warpPerspective(img_filtered, matrix, [width, height])
                img_warped = cv2.resize(img_warped, dsize=(640, 480))

                # Sliding Window로 차선 중심 찾기
                _, L, _ = cv2.split(cv2.cvtColor(img_warped, cv2.COLOR_BGR2HLS))
                _, img_binary = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY)
                img_masked = region_of_interest(img_binary)
                out_img, self.x_location, _ = self.slidewindow.slidewindow(img_masked, True)
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
                    should_stop = False
                    if len(self.obstacle_list_dynamic) > 0:
                        for obstacle in self.obstacle_list_dynamic:
                            # ★★★ 5m 앞에 유효한 장애물이 있으면 즉시 정지 ★★★
                            if (0.0 < obstacle.x < 5.0) and self.is_valid_obstacle(obstacle):
                                print('5m 내 동적 장애물 감지: 즉시 정지!')
                                motor_msg = 0
                                self.publishCtrlCmd(0.0, -radians(angle), 1.0)
                                self.dynamic_stop_flag = True
                                should_stop = True
                                break # 정지 결정했으므로 더 이상 검사할 필요 없음
                    
                    if should_stop:
                        continue # 다음 루프로 넘어가기

                    # 동적 장애물 미션 완료 후 정적 시퀀스로 전환
                    if self.dynamic_stop_flag and len(self.obstacle_list_dynamic) == 0:
                        print("동적 회피 완료!")
                        self.is_dynamic = False # 시퀀스 전환!
                        motor_msg = 15 # 전환 후 빠르게 출발
                    else:
                        motor_msg = 10
                
                # # --- 2. 정적 장애물 회피 시퀀스 ---
                # if not self.is_dynamic:
                #     print("정적 시퀀스로 전환합니다.")
                #     tunnel_static_roi_check_arr = [0, 0, 0]
                #     # 'L' 상태에서 장애물 감지 및 'A' 상태로 전환
                #     if self.static_state == 'L':
                #         print("seq : L")
                #         for obstacle in self.obstacle_list_static:
                #             if 0.5 < obstacle.x < 8.5 and -1.5 <= obstacle.y <= 0.0: # 오른쪽 장애물
                #                 tunnel_static_roi_check_arr[1] = 1
                #             if 0.5 < obstacle.x < 8.5 and 0.0 <= obstacle.y <= 1.5: # 왼쪽 장애물
                #                 tunnel_static_roi_check_arr[0] = 1

                #         if tunnel_static_roi_check_arr == [0, 1, 0]: # 오른쪽 장애물 -> 왼쪽 회피
                #             print("seq : A / R -> L")
                #             self.static_state = 'A'; motor_msg = 7
                #             self.gt_heading = self.real_heading; self.avoid_heading = 25; self.return_heading = -13
                #         elif tunnel_static_roi_check_arr == [1, 0, 0]: # 왼쪽 장애물 -> 오른쪽 회피
                #             print("seq : A / L -> R")
                #             self.static_state = 'A'; motor_msg = 7
                #             self.gt_heading = self.real_heading; self.avoid_heading = -25; self.return_heading = 13

                #     # 'A' 상태에서 IMU 기반 회피 및 'L' 상태 복귀
                #     if self.static_state == 'A':
                #         print("seq : A")
                #         if self.local_heading is not None:
                #             # 복귀 완료 조건
                #             if (self.avoid_heading > 0 and self.local_heading < self.return_heading) or \
                #                (self.avoid_heading < 0 and self.local_heading > self.return_heading):
                #                 self.static_state = 'L'
                #                 self.gt_heading, self.avoid_heading, self.return_heading = None, None, None
                #             else: # 회피/복귀 중
                #                 angle = 1 * abs(self.local_heading - self.return_heading)
                #                 motor_msg = 7
                # --- 2. 정적 장애물 회피 시퀀스 ---
                if not self.is_dynamic:
                    # ------------------------------
                    # 1. 새로운 장애물 구간 탐지 및 회피 시작
                    # 조건: (아직 회피 중이 아님) + (헤딩 기준값 없음)
                    # 동작: 전방 0.5~8.5m, 좌우 -1.5~1.5m 안에 장애물이 있으면 회피 시작
                    # ------------------------------
                    if not self.is_passing_static_zone and self.gt_heading is None:
                        is_obstacle_found = False
                        for obstacle in self.obstacle_list_static:
                            print(f"[DEBUG] 정적 후보 장애물: x={obstacle.x:.2f}, y={obstacle.y:.2f}, dist={obstacle.distance():.2f}")
                            if 0.5 < obstacle.x < 8.5 and -1.5 < obstacle.y < 1.5:
                                is_obstacle_found = True
                                print("[DEBUG] → 정적 장애물 조건 충족! 회피 시작 준비")
                                break
                        
                        if is_obstacle_found:
                            self.is_passing_static_zone = True
                            motor_msg = 7
                            self.gt_heading = self.real_heading
                            self.local_heading = 0.0
                            self.avoid_heading = 25
                            self.return_heading = -13
                            print(f"[DEBUG] 회피 시작: gt_heading={self.gt_heading}, avoid={self.avoid_heading}, return={self.return_heading}")

                    # ------------------------------
                    # 2. IMU 회피 기동 실행
                    # 조건: 회피 동작 중 (gt_heading이 세팅되어 있음)
                    # 동작: local_heading이 return_heading 조건을 만족하면 회피 종료
                    # ------------------------------
                    if self.gt_heading is not None:
                        if self.local_heading is None:
                            print("[DEBUG] local_heading 아직 없음 (IMU 미반영)")
                        else:
                            angle = 1 * abs(self.local_heading - self.return_heading)
                            motor_msg = 7
                            print(f"[DEBUG] 회피 진행 중: local_heading={self.local_heading:.2f}, target={self.return_heading}, angle={angle:.2f}")

                        is_turn_complete = (self.avoid_heading > 0 and self.local_heading < self.return_heading) or \
                                        (self.avoid_heading < 0 and self.local_heading > self.return_heading)
                        
                        if is_turn_complete:
                            print("[DEBUG] 회피 동작 완료 → 헤딩값 초기화")
                            self.gt_heading, self.avoid_heading, self.return_heading = None, None, None

                    # ------------------------------
                    # 3. 장애물 구간 완전 통과 확인
                    # 조건: 회피는 끝났지만, 여전히 구간 플래그가 True일 때
                    # 동작: 전방에 더 이상 장애물이 없으면 구간 종료
                    # ------------------------------
                    if self.is_passing_static_zone and self.gt_heading is None:
                        is_zone_cleared = True
                        for obstacle in self.obstacle_list_static:
                            if obstacle.x > 0.5:  # 아직 앞쪽에 장애물이 남아 있음
                                is_zone_cleared = False
                                print(f"[DEBUG] 아직 구간에 장애물 있음: x={obstacle.x:.2f}, y={obstacle.y:.2f}")
                                break
                        
                        if is_zone_cleared:
                            self.is_passing_static_zone = False
                            print("[DEBUG] 정적 장애물 구간 완전 통과! 플래그 리셋")


                # 최종 제어값 발행
                servo_msg = -radians(angle)
                self.publishCtrlCmd(motor_msg, servo_msg, 0)

            except Exception as e:
                print(e)

            rate.sleep()

    def image_callback(self, msg):
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg)

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
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

if __name__ == "__main__":
    try:
        lanenet_detection_node = LanenetDetection()
    except rospy.ROSInterruptException:
        pass