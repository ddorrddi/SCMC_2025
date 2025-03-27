#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy 
from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage
from std_msgs.msg import Float64, Int64, Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Imu, Image
from geometry_msgs.msg import  Vector3
from move_base_msgs.msg import MoveBaseActionResult
from pyproj import Proj, transform
from nav_msgs.msg import Odometry
from math import pi, sqrt, atan2, radians
from lidar_object_detection.msg import ObjectInfo
from slidewindow import SlideWindow
from visualization_msgs.msg import MarkerArray
from race.msg import drive_values


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

        self.ctrl_cmd_pub = rospy.Publisher('control_value', drive_values, queue_size=1)
        # self.tunnel_static_roi_flag_pub = rospy.Publisher('/tunnel_static_roi_flag', Bool, queue_size=1)

        rospy.Subscriber("/usb_cam/image_raw", Image, self.camCB)
        rospy.Subscriber("/bounding_box", MarkerArray, self.objectCB)
        rospy.Subscriber("/imu", Imu, self.imuCB)
        
        #-------------------------------------------------------------------------------------- #
        self.proj_UTM = Proj(proj='utm', zone = 52, elips='WGS84', preserve_units=False)
        self.bridge = CvBridge()
        self.ctrl_cmd_msg = CtrlCmd()

        self.drive_value = drive_values()
        # self.status_msg   = EgoStatus()
        self.traffic_msg = GetTrafficLightStatus()

        self.ctrl_cmd_msg.longlCmdType = 2
        self.is_gps = True
        self.is_imu = True

        self.slidewindow = SlideWindow()

        self.traffic_flag = 0
        self.prev_signal = 0
        self.signal = 0
        self.stopline_flag = 0
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
        self.left_indices = []
        self.right_indices = []
        self.lidar_obstacle_info = []
        self.gt_heading = 0

        # 640, 480
        # 320
        
        self.heading = 0.0
        self.x_location = 320
        self.last_x_location = 320

        self.prev_center_index = 320
        self.center_index = 320
        self.standard_line = 320
        self.degree_per_pixel = 0
        self.avoid_x_location = 320

        self.current_lane = 2
        self.avoid_status = 'lanekeeping'
        self.sliding_window_select_line = 'Right'
        
        self.current_waypoint = 0

        self.up_hist_end_line = 400
        self.down_hist_start_line = 400

        self.fusion_result = []

        self.is_dynamic_passed = False

        #------------------------------------- 동적 장애물 파라미터 -------------------------------------#
        self.obstacle_list = []
        self.theta_list = []
        self.obstacle_flag = False
        self.brake_cnt = 0
        var_rate = 20

        #-----------------------------------------------------------------------------------------------#

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

        # 트랙바 초기값 설정
        self.lower_yellow = np.array([95, 51, 101])
        self.upper_yellow = np.array([104, 255, 255])
        self.lower_white = np.array([80, 0, 158])
        self.upper_white = np.array([132, 16, 255])

        # 트랙바 창 생성
        cv2.namedWindow("Trackbars")
        self.create_trackbars()

        rate = rospy.Rate(var_rate)  # hz 
        while not rospy.is_shutdown():

            if len(self.img)!= 0:

                y, x = self.img.shape[0:2]

                # -------- 기존의 HSV를 통한 이미지 처리 -------- # 
                self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                
                h, s, v = cv2.split(self.img_hsv)

                # 트랙바로부터 HSV 값 읽기
                self.read_trackbar_values()

                # 노란색 및 흰색 마스크 생성
                mask_yellow = cv2.inRange(self.img_hsv, self.lower_yellow, self.upper_yellow)
                mask_white = cv2.inRange(self.img_hsv, self.lower_white, self.upper_white)
                filtered_yellow = cv2.bitwise_and(self.img, self.img, mask=mask_yellow)
                filtered_white = cv2.bitwise_and(self.img, self.img, mask=mask_white)
                masks = cv2.bitwise_or(mask_yellow, mask_white)
                filtered_img = cv2.bitwise_and(self.img, self.img, mask=masks)

                # ---------------------------------------------------------------- #

                left_margin = 180
                top_margin = 256
                src_point1 = [0, 413]      # 왼쪽 아래
                src_point2 = [left_margin, top_margin]
                src_point3 = [x-left_margin, top_margin]
                src_point4 = [x , 413]  

                src_points = np.float32([src_point1, src_point2, src_point3, src_point4])
                
                dst_point1 = [x//4, 540]    # 왼쪽 아래
                dst_point2 = [x//4, 0]      # 왼쪽 위
                dst_point3 = [x//4*3, 0]    # 오른쪽 위
                dst_point4 = [x//4*3, 540]  # 오른쪽 아래

                dst_points = np.float32([dst_point1, dst_point2, dst_point3, dst_point4])
                
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                self.warped_img = cv2.warpPerspective(filtered_img, matrix, [x, y])

                if self.tunnel_static_half_flag == False:
                    translated_img = self.translate_image(self.warped_img, tx=90, ty=0)
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
                pid = PID(0.2, 0.1, 0.03)

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
                motor_msg = 5

                angle = pid.pid_control(self.center_index - 320)
                # print("angle", angle)
                # servo_msg = -radians(angle)


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
                                # print("Theta: ", theta)
                    
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
                        angle = 0.0028 * (theta**2) - 0.42 * theta - 30.0
                        # servo_msg = -radians(angle)
                        # print("@@@@@@@@@@@@@@@@@@@@@@@")
                        break

                # print("인지 배열: ", tunnel_statlc_roi_check_arr)
                # print("선택 방향: ", self.tunnel_static_flag)

                # print("servo_msg: ", servo_msg)
                self.publishCtrlCmd(motor_msg, angle, 0)


                cv2.imshow('self.img', self.img)
                cv2.imshow("Yellow Mask", filtered_yellow)
                cv2.imshow("White Mask", filtered_white)
                cv2.imshow("Filtered Image", filtered_img)
                cv2.imshow("Warped Image", self.warped_img)
                cv2.imshow("Output Image", self.out_img)

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

    def create_trackbars(self):
        # 노란색 HSV 범위 트랙바
        cv2.createTrackbar("Yellow Lower H", "Trackbars", self.lower_yellow[0], 180, self.nothing)
        cv2.createTrackbar("Yellow Lower S", "Trackbars", self.lower_yellow[1], 255, self.nothing)
        cv2.createTrackbar("Yellow Lower V", "Trackbars", self.lower_yellow[2], 255, self.nothing)
        cv2.createTrackbar("Yellow Upper H", "Trackbars", self.upper_yellow[0], 180, self.nothing)
        cv2.createTrackbar("Yellow Upper S", "Trackbars", self.upper_yellow[1], 255, self.nothing)
        cv2.createTrackbar("Yellow Upper V", "Trackbars", self.upper_yellow[2], 255, self.nothing)

        # 흰색 HSV 범위 트랙바
        cv2.createTrackbar("White Lower H", "Trackbars", self.lower_white[0], 180, self.nothing)
        cv2.createTrackbar("White Lower S", "Trackbars", self.lower_white[1], 255, self.nothing)
        cv2.createTrackbar("White Lower V", "Trackbars", self.lower_white[2], 255, self.nothing)
        cv2.createTrackbar("White Upper H", "Trackbars", self.upper_white[0], 180, self.nothing)
        cv2.createTrackbar("White Upper S", "Trackbars", self.upper_white[1], 255, self.nothing)
        cv2.createTrackbar("White Upper V", "Trackbars", self.upper_white[2], 255, self.nothing)

    def read_trackbar_values(self):
        self.lower_yellow[0] = cv2.getTrackbarPos("Yellow Lower H", "Trackbars")
        self.lower_yellow[1] = cv2.getTrackbarPos("Yellow Lower S", "Trackbars")
        self.lower_yellow[2] = cv2.getTrackbarPos("Yellow Lower V", "Trackbars")
        self.upper_yellow[0] = cv2.getTrackbarPos("Yellow Upper H", "Trackbars")
        self.upper_yellow[1] = cv2.getTrackbarPos("Yellow Upper S", "Trackbars")
        self.upper_yellow[2] = cv2.getTrackbarPos("Yellow Upper V", "Trackbars")

        self.lower_white[0] = cv2.getTrackbarPos("White Lower H", "Trackbars")
        self.lower_white[1] = cv2.getTrackbarPos("White Lower S", "Trackbars")
        self.lower_white[2] = cv2.getTrackbarPos("White Lower V", "Trackbars")
        self.upper_white[0] = cv2.getTrackbarPos("White Upper H", "Trackbars")
        self.upper_white[1] = cv2.getTrackbarPos("White Upper S", "Trackbars")
        self.upper_white[2] = cv2.getTrackbarPos("White Upper V", "Trackbars")

    def nothing(self, x):
        pass

    def camCB(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg)

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
        self.drive_value.throttle = motor_msg
        self.drive_value.steering = servo_msg
        self.drive_value.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.drive_value)

if __name__ == "__main__":
    try:
        lane_detection_node = LaneDetection()
    except rospy.ROSInterruptException:
        pass
