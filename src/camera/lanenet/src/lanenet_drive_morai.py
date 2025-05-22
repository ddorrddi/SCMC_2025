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
        self.is_detected = True
        self.current_lane = "LEFT"
        self.last_angle = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = '/home/foscar/iscc_2024/src/camera/lanenet/src/lanenet_.model'
        self.LaneNet_model = Lanenet(2, 4)
        self.LaneNet_model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        self.LaneNet_model.to(self.device)
        self.heading = 0.0


        self.motor_msg = 10

        self.x_location = 320
        self.last_x_location = 320
        #------------------------------------- 동적 장애물 파라미터 -------------------------------------#

        self.is_dynamic = False
        self.dynamic_stop_flag = False
        self.obstacle_list_dynamic = []
        self.theta_list = []
        var_rate = 20

        self.tunnel_dynamic_steering_list = []
        self.tunnel_dynamic_steering_list_max_len = 50
        self.steer_diff = -1
        self.steer_diff_threshold = 5
        self.is_steer_stable = False

        self.tunnel_dynamic_obstacle_y_list = []
        self.tunnel_dynamic_obstacle_y_list_max_len = 20
        self.obstacle_y_diff = -1
        self.obstacle_y_diff_threshold = 0.4
        self.z_change_threshold = 0.1
        self.volume_change_threshold = 0.1
       

        #------------------------------------- 정적 장애물 파라미터 --------------------------------------#
        tunnel_statlc_roi_check_arr = [0, 0, 0]


        # state
        # 'L': Lane
        # 'A': Avoid
        # 'R': Return   
        self.static_state = 'L'
        
        self.real_heading = None
        self.gt_heading = None

        self.avoid_heading = None
        self.return_heading = None

        self.local_heading = None

        self.global_path_finish = False
        
        self.obstacle_list_static = []

        rospy.Subscriber("/global_path_finish", Bool, self.global_path_finish_CB)
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback)
        rospy.Subscriber("imu", Imu, self.imuCB)
        rospy.Subscriber("/bounding_box_static", MarkerArray, self.objectCB_static)
        rospy.Subscriber("/bounding_box_dynamic", MarkerArray, self.objectCB_dynamic)
        rate = rospy.Rate(var_rate)
        
        while not rospy.is_shutdown():

            pid = PID(0.015, 0.003, 0.010)
            try:
                
                cv2.imshow("lane_image", self.image)

                show_img = self.inference(self.image)
                # init_show_img(show_img)

                # cv2.imshow("show", show_img)


                img_frame = show_img.copy() # img_frame변수에 카메라 이미지를 받아옵니다.  
                height,width,channel = img_frame.shape # 이미지의 높이,너비,채널값을 변수에 할당합니다.
               
               
                img_roi = img_frame[280:470,0:]   # y좌표 0~320 사이에는 차선과 관련없는 이미지들이 존재하기에 노이즈를 줄이기 위하여 roi설정을 해주었습니다.



                img_filtered = color_filter(img_roi)   #roi가 설정된 이미지를 color_filtering 하여 흰색 픽셀만을 추출해냅니다.

                height, width, channel = img_filtered.shape


                # cv2.imshow("filter", img_filtered)
               
               
                # img_warped = bird_eye_view(img_filtered,width,height) # 앞서 구현한 bird-eye-view 함수를 이용하여 시점변환해줍니다.

                left_margin = 195
                top_margin =  44
                src_point1 = [0, 190]      # 왼쪽 아래
                src_point2 = [left_margin, top_margin]
                src_point3 = [width-left_margin, top_margin]
                src_point4 = [width , 190]  

                src_points = np.float32([src_point1, src_point2, src_point3, src_point4])
               
                dst_point1 = [width//4, 190]    # 왼쪽 아래
                dst_point2 = [width//4, 0]      # 왼쪽 위
                dst_point3 = [width//4*3, 0]    # 오른쪽 위
                dst_point4 = [width//4*3, 190]  # 오른쪽 아래

                dst_points = np.float32([dst_point1, dst_point2, dst_point3, dst_point4])
               
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                img_warped = cv2.warpPerspective(img_filtered, matrix, [width, height])


                img_warped = cv2.resize(img_warped,dsize=(640,480))
                # cv2.imshow('img_warped', img_warped)
           
           

                #정적 장애물 회피 전(오른쪽 붙어 주행)
                if self.static_state == 'L':
                    translated_img = self.translate_image(img_warped, tx=35, ty=0)
                #정적 장애물 회피 시작(왼쪽 붙어 주행)
                elif (self.static_state != 'L') and (self.is_dynamic == False):
                    translated_img = self.translate_image(img_warped, tx=-35, ty=0)
                #정적 장애물 회피 종료 & 동적 시작(가운데 주행)
                elif self.is_dynamic == True:
                    translated_img = img_warped
           
                _, L, _ = cv2.split(cv2.cvtColor(translated_img, cv2.COLOR_BGR2HLS))
                _, img_binary = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY) 

                img_masked = region_of_interest(img_binary) #이진화까지 마친 이미지에 roi를 다시 설정하여줍니다.
                # cv2.imshow("img_masked", img_masked)obstacle2_list

                out_img, self.x_location, current_lane = self.slidewindow.slidewindow(img_masked, self.is_detected)
                if self.x_location is None:
                    self.x_location = self.last_x_location
                else:
                    self.last_x_location = self.x_location
               

                img_masked_colored = cv2.cvtColor(img_masked, cv2.COLOR_GRAY2BGR)
           
                if out_img.shape == img_masked_colored.shape:
                    img_blended = cv2.addWeighted(out_img, 1, img_masked_colored, 0.6, 0)  # sliding window 결과를 시각화하기 위해 out_img와 시점 변환된 이미지를 merging 하였습니다.
                    cv2.imshow('img_blended', img_blended)
                else:
                    print(f"Shape mismatch: out_img {out_img.shape}, img_masked {img_masked_colored.shape}")
               
                #동적 회피 까지 종료 됐다면 빠르게 가기!
                if self.dynamic_stop_flag == True and len(self.obstacle_list_dynamic) == 0:
                    self.motor_msg = 15
                else:
                    self.motor_msg = 10
                
                angle = pid.pid_control(self.x_location - 320)
                
                cv2.waitKey(1)

                if not self.global_path_finish:
                    print("global path 작동중")
                    rate.sleep()
                    continue
                
                # ---------------------- 정적 회피 주행 ---------------------- #
                if self.is_dynamic == False:
                    tunnel_statlc_roi_check_arr = [0, 0, 0]
                    self.theta_list = []
                    # 터널 정적 PE-Drum s자 회피
                    if len(self.obstacle_list_static) > 0:
                        last_tunnel_static_obstacle = None
                        for obstacle in self.obstacle_list_static:

                            if self.static_state == 'L':
                                # 전방 체크
                                if 0.5 < obstacle.x < 8.5 and -1.5 <= obstacle.y <= 1.5:
                                    tunnel_statlc_roi_check_arr[1] = 1
                                    self.motor_msg = 10
                                    print("인식")
                                    self.static_state = 'R'
                                    self.avoid_heading = 35
                                    self.return_heading = -17

                                # 왼쪽뒤 체크
                        #         if -6 <= obstacle.x <= -0.5 and 0 < obstacle.y <= 3.5:
                        #             tunnel_statlc_roi_check_arr[0] = 1
                        #             self.motor_msg = 7
                        #         # 시작 타이밍 체크
                        # if tunnel_statlc_roi_check_arr == [0, 1, 0]:
                        #     for obstacle in self.obstacle_list_static:
                        #         if 1.0 < obstacle.x < 8.0:
                        #             tunnel_statlc_roi_check_arr[2] = 1
                        #             # flag
                        #             self.static_state = 'R'
                        #             self.motor_msg = 7
                        #             # gt heading
                        #             self.gt_heading = self.real_heading

                        #             self.avoid_heading = 30
                        #             self.return_heading = -17

                        #             break
                       
                    # 터널 내 장애물 회피 각도 조정
                    if self.static_state == 'A':
                        if self.local_heading != None:
                            if (self.avoid_heading > self.local_heading): # 목표 heading에 도달하지 못했으면 좌조향
                                angle = -0.7 * abs(self.local_heading - self.avoid_heading)
                            else:
                                self.static_state = 'R'
                    
                    elif self.static_state == 'R':

                        if self.local_heading != None:
                            if (self.return_heading < self.local_heading): # 목표 heading에 도달하지 못했으면 우조향
                                angle = 1 * abs(self.local_heading - self.return_heading)
                            else:
                                self.gt_heading = None
                                self.avoid_heading = None
                                self.return_heading = None

                # ---------------------- 정적 회피 주행 end ---------------------- #

                # 최종 계산된 조향각의 변화량을 비교하기 때문에 이 동적 판단 코드는 이 위치가 맞음.
                # 매번 조향각 추가
                self.tunnel_dynamic_steering_list.append(angle)
                if len(self.tunnel_dynamic_steering_list) > self.tunnel_dynamic_steering_list_max_len:
                    self.tunnel_dynamic_steering_list.pop(0)

               
                # print('self.tunnel_dynamic_steering_list', self.tunnel_dynamic_steering_list)
                # len == 5
                for i in range(len(self.tunnel_dynamic_steering_list) - 1): # 0 ~ 3
                    self.steer_diff = abs(self.tunnel_dynamic_steering_list[-1] - self.tunnel_dynamic_steering_list[i])
                    if self.steer_diff > self.steer_diff_threshold:
                        self.is_steer_stable = False
                        break
                else:
                    self.is_steer_stable = True


                # print("flag list", tunnel_statlc_roi_check_arr)
                # print("정적 회피 시작: ", self.static_state)
                # print("local_heading", self.local_heading)
                # print("gt_heading", self.gt_heading)
                # print("real_heading", self.real_heading)
                # print("steer_stable", self.is_steer_stable)
                print("static_state",self.static_state)
                # 동적 flag 올리기 위함 
                # self.static_state + steering(변화량이 적을때)  -> is_dynamic(True)
                if (self.static_state == 'R') and (self.is_steer_stable == True):
                    self.is_dynamic = True


                # MORAI 상에서는 -radian을 씌워야 하는데 해당 위치에서 해야 한번에 가능
                servo_msg = -radians(angle)

                # 정적 상황이 끝난 뒤부터는 계속 들어감
                if self.is_dynamic == True:
                    # Obstacle 이 있을 때 서행 (2~3 정도로 엄청 느리게)
                    if len(self.obstacle_list_dynamic) > 0:
                        for obstacle in self.obstacle_list_dynamic:
                            if (0.0 < obstacle.x < 7.0) and (-4.0 < obstacle.y < 4.0):
                                self.motor_msg = 5
                                if self.is_valid_obstacle(obstacle):
                                    # 영역 안 장애물의  y 리스트를 갱신
                                    self.tunnel_dynamic_obstacle_y_list.append(obstacle.y)
                                    print("동적 장애물: ", obstacle.y)
                                    if len(self.tunnel_dynamic_obstacle_y_list) > self.tunnel_dynamic_obstacle_y_list_max_len:
                                        self.tunnel_dynamic_obstacle_y_list.pop(0)
                    else:
                        self.tunnel_dynamic_obstacle_y_list = []

                    # Obstacle 이 움직일 때 멈추기
                    if len(self.tunnel_dynamic_obstacle_y_list) >= 1:
                        if -1.0 < self.tunnel_dynamic_obstacle_y_list[-1] < 1.0:
                            print('정지!!!')
                            self.motor_msg = 0
                            self.publishCtrlCmd(0.0, servo_msg, 1.0)
                            self.dynamic_stop_flag = True
                            continue
                    
                print(self.motor_msg)
                self.publishCtrlCmd(self.motor_msg, servo_msg, 0)
                # print("장애물 부피: ", obstacle.volume())

                print()

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
        
        # 장애물을 거리순으로 정렬
        self.obstacle_list_static.sort(key=lambda obstacle: obstacle.distance())    

    def objectCB_dynamic(self, msg):
        self.obstacle_list_dynamic = []
       
        for marker in msg.markers:
            width = marker.scale.x
            height = marker.scale.y
            depth = marker.scale.z

            obstacle = Obstacle_Dynamic(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z, width, height, depth)
           
            if len(self.obstacle_list_dynamic) > 0:
                # 리스트에 있는 장애물들의 평균 부피와 z 값을 계산
                avg_volume = sum([obs.volume() for obs in self.obstacle_list_dynamic]) / len(self.obstacle_list_dynamic)
                avg_z = sum([obs.z for obs in self.obstacle_list_dynamic]) / len(self.obstacle_list_dynamic)

                # 새로운 장애물의 부피와 z 값을 기존의 평균과 비교
                if (0.8 * avg_volume <= obstacle.volume() <= 1.2 * avg_volume) and (0.8 * avg_z <= obstacle.z <= 1.2 * avg_z):
                    self.obstacle_list_dynamic.append(obstacle)
            else:
                self.obstacle_list_dynamic.append(obstacle)
       
        # 장애물을 거리순으로 정렬
        self.obstacle_list_dynamic.sort(key=lambda obstacle: obstacle.distance())
        
    def global_path_finish_CB(self, msg: Bool):
        self.global_path_finish = msg.data

    def imuCB(self, msg):
        # 쿼터니언을 사용하여 roll, pitch, yaw 계산
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )

        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll, pitch, yaw = euler

        self.real_heading = yaw * 180.0 / math.pi

        if self.gt_heading != None:
            self.local_heading = self.real_heading - self.gt_heading
            if self.local_heading > 180:
                self.local_heading -= 360
            elif self.local_heading < -180:
                self.local_heading += 360
        else:
            self.local_heading = None

    # def is_valid_obstacle(self, obstacle):
        
    #     # 현재 장애물의 부피와 z 값을 가져옴
    #     current_volume = obstacle.volume()
    #     current_z = obstacle.z

    #     # 기존 장애물들의 부피와 z 값의 평균을 계산
    #     avg_volume = sum(obs.volume() for obs in self.obstacle_list_dynamic) / len(self.obstacle_list_dynamic)
    #     avg_z = sum(obs.z for obs in self.obstacle_list_dynamic) / len(self.obstacle_list_dynamic)

    #     # 부피와 z 값이 평균의 80%~120% 범위 내에 있는지 확인
    #     volume_within_range = (0.8 * avg_volume <= current_volume <= 1.2 * avg_volume)
    #     z_within_range = (0.8 * avg_z <= current_z <= 1.2 * avg_z)

    #     # 부피와 z 값이 모두 기준 내에 있으면 유효한 장애물로 판단
    #     return volume_within_range and z_within_range
   

    #상대적 변화율을 이용한 volume, z 보정 적용 함수, threshold값 임의로 0.1로 세팅했음 테스트 요.
    def is_valid_obstacle(self, obstacle):
        # 현재 장애물의 부피와 z 값을 가져옴
        current_volume = obstacle.volume()
        current_z = obstacle.z

        # 기존 장애물들의 평균 부피와 z 값 계산
        avg_volume = sum(obs.volume() for obs in self.obstacle_list_dynamic) / len(self.obstacle_list_dynamic)
        avg_z = sum(obs.z for obs in self.obstacle_list_dynamic) / len(self.obstacle_list_dynamic)

        # 상대적 변화율 계산
        z_change_rate = abs(current_z - avg_z) / avg_z
        volume_change_rate = abs(current_volume - avg_volume) / avg_volume

        # 변화율이 특정 임계값 이내에 있는지 확인
        z_within_range = z_change_rate <= self.z_change_threshold
        volume_within_range = volume_change_rate <= self.volume_change_threshold

        return z_within_range and volume_within_range


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

    def inference(self, gt_img_org):
        # BGR 순서
        org_shape = gt_img_org.shape
        gt_image = cv2.resize(gt_img_org, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
        gt_image = gt_image / 127.5 - 1.0
        gt_image = torch.tensor(gt_image, dtype=torch.float)
        gt_image = np.transpose(gt_image, (2, 0, 1))
        gt_image = gt_image.to(self.device)
        # lane segmentation
        binary_final_logits, instance_embedding = self.LaneNet_model(gt_image.unsqueeze(0))
        binary_final_logits, instance_embedding = binary_final_logits.to('cpu'), instance_embedding.to('cpu')
        binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().numpy()
        binary_img[0:65,:] = 0 #(0~85행)을 무시 - 불필요한 영역 제거      
        binary_img=binary_img.astype(np.uint8)
        binary_img[binary_img>0]=255
        # 차선 클러스터링, 색상 지정
        # rbg_emb, cluster_result = process_instance_embedding(instance_embedding, binary_img,distance=1.5, lane_num=2)
        # rbg_emb = cv2.resize(rbg_emb, dsize=(org_shape[1], org_shape[0]), interpolation=cv2.INTER_LINEAR)
        # a = 0.1
        # frame = a * gt_img_org[..., ::-1] / 255 + rbg_emb * (1 - a)
        # frame = np.rint(frame * 255)
        # frame = frame.astype(np.uint8)

        frame = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        frame = cv2.resize(frame, dsize=(640, 480))

        return frame
   
    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        # if not self.global_path_finish:
        #     return
        #print("lanenet주행")
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

if __name__ == "__main__":
    try:
        lanenet_detection_node = LanenetDetection()
    except rospy.ROSInterruptException:
        pass
