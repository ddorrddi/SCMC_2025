#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy 
from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage
from std_msgs.msg import Float64, Int64, Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import  Vector3
from move_base_msgs.msg import MoveBaseActionResult
from pyproj import Proj, transform
from nav_msgs.msg import Odometry
from math import pi, sqrt, atan2, radians
from lidar_object_detection.msg import ObjectInfo
from slidewindow import SlideWindow
from visualization_msgs.msg import MarkerArray

import tf
import time
import cv2
import numpy as np
import math

class Obstacle:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


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

class LaneDetection:
    def __init__(self):
        rospy.init_node("lane_detection_node")

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.camCB)
        rospy.Subscriber("/bounding_box", MarkerArray, self.objectCB)
        rospy.Subscriber("/imu", Imu, self.imuCB)

        self.ctrl_cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        # self.tunnel_static_roi_flag_pub = rospy.Publisher('/tunnel_static_roi_flag', Bool, queue_size=1)

        
        #-------------------------------------------------------------------------------------- #
        self.proj_UTM = Proj(proj='utm', zone = 52, elips='WGS84', preserve_units=False)
        self.bridge = CvBridge()
        self.ctrl_cmd_msg = CtrlCmd()
        # self.status_msg   = EgoStatus()
        self.traffic_msg = GetTrafficLightStatus()

        self.ctrl_cmd_msg.longlCmdType = 2


        self.slidewindow = SlideWindow()

        self.img = []
        self.warped_img = []
        self.grayed_img = []
        self.out_img = []
        self.yellow_img = []
        self.white_img = []
        self.img_hsv = []
        self.h = []
        self.s = []
        self.v = []
        self.bin_img = []

        
        self.heading = 0.0
        self.x_location = 320
        self.last_x_location = 320

        self.prev_center_index = 320
        self.center_index = 320
        self.standard_line = 320
        self.degree_per_pixel = 0
        self.avoid_x_location = 320



        #------------------------------------- 동적 장애물 파라미터 -------------------------------------#
        self.is_dynamic_passed = False

        self.obstacle_list = []
        self.theta_list = []
        self.obstacle_flag = False
        self.brake_cnt = 0
        var_rate = 20

        #------------------------------------- 정적 장애물 파라미터 --------------------------------------#

        self.static_obstacle = False
        self.static_left_done = False

        self.avg_y_early_detected_obstacle = 0.0
        self.avg_y_late_detected_obstacle = 0.0

        self.not_detected = 0

        self.tunnel_static_flag = 0
        self.passed_half_tunnel_static_keeping = False

        self.tunnel_static_roi_flag = False
        tunnel_statlc_roi_check_arr = [0, 0, 0]

        self.tunnel_static_half_flag = False

        rate = rospy.Rate(var_rate)  # hz 
        while not rospy.is_shutdown():

            if len(self.img)!= 0:

                y, x = self.img.shape[0:2]

                # -------- 기존의 HSV를 통한 이미지 처리 -------- # 
                self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                
                h, s, v = cv2.split(self.img_hsv)

                # 터널 밖 노란 선 hsv
                yellow_lower_1 = np.array([15, 75, 180])
                yellow_upper_1 = np.array([23, 150, 245])
                yellow_range_1 = cv2.inRange(self.img_hsv, yellow_lower_1, yellow_upper_1)

                # 터널 안 노란 선 hsv
                yellow_lower_2 = np.array([24, 110, 50])
                yellow_upper_2 = np.array([90, 180, 85])
                yellow_range_2 = cv2.inRange(self.img_hsv, yellow_lower_2, yellow_upper_2)
                
                # 터널 밖 흰 선 hsv
                white_lower_bound1 = np.array([10, 27, 210])
                white_upper_bound1 = np.array([25, 45, 255])
                white_mask1 = cv2.inRange(self.img_hsv, white_lower_bound1, white_upper_bound1)
                
                # 터널 안 흰 선 hsv
                white_lower_bound2 = np.array([90, 40, 60])
                white_upper_bound2 = np.array([110, 70, 90])
                white_mask2 = cv2.inRange(self.img_hsv, white_lower_bound2, white_upper_bound2)

                self.yellow_range = cv2.bitwise_or(yellow_range_1, yellow_range_2)
                self.white_range = cv2.bitwise_or(white_mask1, white_mask2)
                
                combined_range = cv2.bitwise_or(self.yellow_range, self.white_range)
                filtered_img = cv2.bitwise_and(self.img, self.img, mask=combined_range)

                # ---------------------------------------------------------------- #

                left_margin = 230
                top_margin = 296
                src_point1 = [0, 443]      # 왼쪽 아래
                src_point2 = [left_margin, top_margin]
                src_point3 = [x-left_margin, top_margin]
                src_point4 = [x , 443]  

                src_points = np.float32([src_point1, src_point2, src_point3, src_point4])
                
                dst_point1 = [x//4, 480]    # 왼쪽 아래
                dst_point2 = [x//4, 0]      # 왼쪽 위
                dst_point3 = [x//4*3, 0]    # 오른쪽 위
                dst_point4 = [x//4*3, 480]  # 오른쪽 아래

                dst_points = np.float32([dst_point1, dst_point2, dst_point3, dst_point4])
                
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                self.warped_img = cv2.warpPerspective(filtered_img, matrix, [x, y])
                
                if self.tunnel_static_half_flag == False:
                    translated_img = self.translate_image(self.warped_img, tx=70, ty=0)
                
                elif self.tunnel_static_half_flag == True:
                    translated_img = self.warped_img


                # else:  # 0
                #     translated_img = self.warped_img

                # 기존 HSV 방식에서 다시 살리기
                self.grayed_img = cv2.cvtColor(translated_img, cv2.COLOR_BGR2GRAY)
                
                # 이미지 이진화
                self.bin_img = np.zeros_like(self.grayed_img)
                self.bin_img[self.grayed_img > 20] = 1
                # self.bin_img[self.grayed_img > 150] = 1  # > 150
                
                histogram_y = np.sum(self.bin_img, axis=1)

                
                self.out_img, self.x_location, _ = self.slidewindow.slidewindow(self.bin_img, self.tunnel_static_flag)


                pid = PID(0.015, 0.003, 0.010)

                if self.x_location is None:
                    self.x_location = self.last_x_location
                else:
                    self.last_x_location = self.x_location
                
                self.standard_line = x // 2
                self.degree_per_pixel = 1 / x
                self.prev_center_index = self.center_index
                self.center_index = self.x_location

                # if self.tunnel_static_flag == 2:
                #     motor_msg = 1
                # else:
                motor_msg = 8

                angle = pid.pid_control(self.center_index - 320)
                # print("angle", angle)
                servo_msg = -radians(angle)


                tunnel_statlc_roi_check_arr = [0, 0]
                self.theta_list = []
                # 터널 정적 PE-Drum s자 회피
                if len(self.obstacle_list) > 0:
                    last_tunnel_static_obstacle = None

                    for obstacle in self.obstacle_list:

                        if self.tunnel_static_half_flag == False:
                            # 전방 체크
                            if 0.5 < obstacle.x < 4.5 and -2 <= obstacle.y <= 2:
                                tunnel_statlc_roi_check_arr[1] = 1
                            # 왼쪽뒤 체크
                            if -6 <= obstacle.x <= -0.5 and 0 < obstacle.y <= 3.5:
                                tunnel_statlc_roi_check_arr[0] = 1

                            if tunnel_statlc_roi_check_arr == [1, 1]:
                                self.tunnel_static_half_flag = True
                                distance_between_obstacle = obstacle.distance()
                                print(distance_between_obstacle)
                    

                        if self.tunnel_static_half_flag == True:
                            
                            if (-0.7 <= obstacle.x <= 5.0) and (-3.0 <= obstacle.y <= 0.5):
                                last_tunnel_static_obstacle = obstacle

                                #장애물과 라이다의 상대각도 계산
                                theta = math.degrees(math.atan2(last_tunnel_static_obstacle.y, last_tunnel_static_obstacle.x))
                                self.theta_list.append(theta)
                                print("Theta: ", theta)
                  
                # 터널 내 장애물 회피 각도 조정
                for theta in self.theta_list:
                    # 첫번째 장애물과 두번째 장애물을 분리 하려는 조건문: 두번째것만 보기 위함.

                    # 6미터 -100 < theta < 5:

                    # 4미터

                    if theta < -100:
                        break
                    
                    if -100 < theta < 5:
                        # 6미터 되는 함수
                        # angle = 0.0027 * (theta**2) - 0.25 * theta - 28.2

                        # 4미터 되는 함수 
                        # angle = 0.0027 * (theta**2) - 0.29 * theta - 28.2
                        angle = 0.0028 * (theta**2) - 0.42 * theta - 30.0
                        servo_msg = -radians(angle)
                        print("@@@@@@@@@@@@@@@@@@@@@@@")
                        break
                # ----------------------- 동적 장애물 (영역 기반) ---------------------------------- #
                # 장애물 감지 감속
                # if len(self.obstacle_list) > 0:

                #     obstacle = self.obstacle_list[0]

                #     self.not_detected = 0
                #     self.AEB = False

                #     if (-3.14 < obstacle.y < 3.14) and (self.is_dynamic_passed == False):
                        
                #         if 1 <= self.brake_cnt <= 5:
                #             self.avg_y_early_detected_obstacle += obstacle.y
                #         elif 395 <= self.brake_cnt <= 399:
                #             self.avg_y_late_detected_obstacle += obstacle.y

                #         self.brake_cnt += 1
                #         self.publishCtrlCmd(0.0, servo_msg, 1.0)
                        
                #         if self.brake_cnt > var_rate * 20: 
                #             delta_y_obstacle = abs((self.avg_y_early_detected_obstacle/5) - (self.avg_y_late_detected_obstacle/5))

                #             print("인지 초기단계 y값 평균: ", self.avg_y_early_detected_obstacle/5)
                #             print("인지 후반단계 y값 편균: ", self.avg_y_late_detected_obstacle/5)
                #             print("그 차이: ", delta_y_obstacle)

                #             if delta_y_obstacle < 1.0: # 정적
                #                 self.static_obstacle = True
                            
                #             else: # 동적이 이상하게 서있는것
                #                 self.is_dynamic_passed = True

                #         self.AEB = True
                        
                    
                #     elif -8.0 < obstacle.y < 8.0:
                #         motor_msg = 12
                            


                #     if (self.AEB == True) and (self.static_obstacle == False):
                #         continue
                # 동적, 정적 구분은 항상 해야만 할듯....
                # y변화량으로만 판단하는게 안전할까...
                # 더 유연한 판단/제어 필요

                else:
                    self.not_detected += 1
                    self.brake_cnt = 0
                    self.avg_y_early_detected_obstacle = 0.0
                    self.avg_y_late_detected_obstacle = 0.0

                if self.not_detected > 10:
                    self.is_dynamic_passed = False
                # print("인지 배열: ", tunnel_statlc_roi_check_arr)
                # print("선택 방향: ", self.tunnel_static_flag)

                # print("servo_msg: ", servo_msg)
                self.publishCtrlCmd(motor_msg, servo_msg, 0)


                cv2.imshow('filtered_img', filtered_img)

                cv2.imshow("out_img", self.out_img)
                cv2.waitKey(1)

            rate.sleep()

    def translate_image(self, image, tx, ty):
        """
        이미지 평행이동 함수
        :param image: 입력 이미지
        :param tx: x축 평행이동 거리
        :param ty: y축 평행이동 거리
        :return: 평행이동된 이미지
        """
        rows, cols = image.shape[:2]
        
        # 평행이동 행렬 생성
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # 이미지 평행이동
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
        
        return translated_image

    def camCB(self, msg):
        self.img = self.bridge.compressed_imgmsg_to_cv2(msg)

    def objectCB(self, msg):
        self.obstacle_list = []
        for marker in msg.markers:
            obstacle = Obstacle(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
            self.obstacle_list.append(obstacle)
        
        # 장애물을 거리순으로 정렬
        self.obstacle_list.sort(key=lambda obstacle: obstacle.distance())
    
    def imuCB(self, msg):
        # 쿼터니언을 사용하여 roll, pitch, yaw 계산
        orientation_q = msg.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
        self.heading = yaw * 180.0 / pi

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

if __name__ == "__main__":
    try:
        lane_detection_node = LaneDetection()
    except rospy.ROSInterruptException:
        pass