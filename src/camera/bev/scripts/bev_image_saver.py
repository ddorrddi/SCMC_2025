#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import os
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class BEVImageSaver:
    def __init__(self):
        rospy.init_node('bev_image_saver', anonymous=True)
        rospy.loginfo("BEV Image Saver Node Started")
        
        self.bridge = CvBridge()
        
        # 이미지 저장 폴더 설정
        self.save_folder = rospy.get_param('~save_folder', '/home/foscar/bev_captured_images2')
        self.create_save_folder()
        
        # 캡쳐 간격 설정 (기본 0.5초)
        self.capture_interval = rospy.get_param('~capture_interval', 0.3)
        
        # 마지막 캡쳐 시간
        self.last_capture_time = 0
        
        # Subscriber 설정
        self.image_sub = rospy.Subscriber('/bev_image', Image, self.image_callback, queue_size=1)
        
        rospy.loginfo(f"이미지 저장 폴더: {self.save_folder}")
        rospy.loginfo(f"캡쳐 간격: {self.capture_interval}초")
        rospy.loginfo("BEV 이미지 자동 캡쳐 시작...")
        
    def create_save_folder(self):
        """저장 폴더 생성"""
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            rospy.loginfo(f"저장 폴더 생성: {self.save_folder}")
    
    def image_callback(self, msg):
        """BEV 이미지 수신 및 저장"""
        current_time = rospy.get_time()
        
        # 캡쳐 간격 체크
        if current_time - self.last_capture_time < self.capture_interval:
            return
            
        try:
            # ROS 이미지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 파일명 생성 (타임스탬프 사용)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
            filename = f"bev_image_{timestamp}.png"
            filepath = os.path.join(self.save_folder, filename)
            
            # 이미지 저장
            cv2.imwrite(filepath, cv_image)
            
            # 로그 출력
            rospy.loginfo(f"BEV 이미지 저장: {filename}")
            
            # 마지막 캡쳐 시간 업데이트
            self.last_capture_time = current_time
            
        except Exception as e:
            rospy.logerr(f"이미지 저장 중 오류: {e}")

if __name__ == '__main__':
    try:
        saver = BEVImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("BEV Image Saver 종료")
    except KeyboardInterrupt:
        rospy.loginfo("사용자에 의해 중단됨")