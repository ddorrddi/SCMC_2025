#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class BEVParkingDetector:
    def __init__(self):
        rospy.init_node('bev_parking_detector_node', anonymous=True)
        rospy.loginfo("BEV Parking Detector Node Started")

        self.bridge = CvBridge()

        # 1단계에서 계산했던 고정 Homography 행렬
        self.H = np.array([

            [-9.49144753e-02, -1.37770938e+00, 3.60924395e+02],
            [ 1.45962627e-02, -9.91594887e-01,  2.69889792e+02],
            [ 7.34354190e-05, -4.27074219e-03,  1.00000000e+00]

        ])
        
        # === [추가] === 캘리브레이션을 통해 얻은 실제 스케일 값
        self.METERS_PER_PIXEL = 0.05

        # Publisher 설정
        self.image_pub = rospy.Publisher("/bev_image", Image, queue_size=1)
        # === [수정] === 최종 미터 단위 경로를 발행할 Publisher
        self.path_meter_pub = rospy.Publisher('/parking_path_meter', Path, queue_size=1)
        
        # Subscriber 설정
        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback, queue_size=1)
        rospy.loginfo("Waiting for images on topic: /image_jpeg/compressed")

    # === [추가] === 픽셀을 차량 로컬 좌표(m)로 변환하는 헬퍼 함수
    def convert_pixel_to_local(self, point_px, image_shape):
        img_h, img_w = image_shape[:2]
        image_center_px = (img_w // 2, img_h // 2)
        
        # 이미지 중앙을 원점으로 하는 픽셀 좌표계로 변환
        delta_px_x = point_px[0] - image_center_px[0]
        delta_px_y = point_px[1] - image_center_px[1]

        # 실제 미터 단위 좌표로 변환
        # 차량 전방(+X) = 이미지 상단(-Y)
        real_x = -delta_px_y * self.METERS_PER_PIXEL
        # 차량 좌측(+Y) = 이미지 좌측(-X)
        real_y = -delta_px_x * self.METERS_PER_PIXEL
        
        return (real_x, real_y)
    
    def get_all_coordinates(self, point_px, image_shape):
        img_h, img_w = image_shape[:2]
        image_center_px = (img_w // 2, img_h // 2)
        
        # 1. 이미지 중앙을 원점으로 하는 픽셀 좌표 (변환된 픽셀 좌표)
        delta_px = (point_px[0] - image_center_px[0], point_px[1] - image_center_px[1])

        # 2. 실제 미터 단위 좌표 (차량 미터 좌표)
        real_x = -delta_px[1] * self.METERS_PER_PIXEL
        real_y = -delta_px[0] * self.METERS_PER_PIXEL
        meter_coords = (real_x, real_y)
        
        return point_px, delta_px, meter_coords

    def image_callback(self, ros_image_compressed):
        try:
            np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
            original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if original_image is None: return
            
            # 원본 이미지의 불필요한 부분 제거 (0으로 채우기)
            height, width, _ = original_image.shape
            original_image[0:270, :] = 0

            # original_image[:, 0:width//2] = 0

            # original_image[350:480,:]=0
        except Exception as e:
            rospy.logerr(f"Image decoding error: {e}")
            return
        # cv2.imshow("1",original_image)
        # --- BEV 변환 로직 ---
        img1_height, img1_width, _ = original_image.shape

        corners_img1 = np.array([[0,0],[img1_width,0],[img1_width,img1_height],[0,img1_height]], dtype='float32')
        warped_corners = cv2.perspectiveTransform(corners_img1[None], self.H)[0]

        ref_width, ref_height = 640, 480
        reference_corners = np.array([[0,0],[ref_width,0],[ref_width,ref_height],[0,ref_height]], dtype='float32')

        all_corners = np.vstack((warped_corners, reference_corners))
        [x_min, y_min] = np.int32(all_corners.min(axis=0))
        [x_max, y_max] = np.int32(all_corners.max(axis=0))

        trans_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        final_H = trans_mat @ self.H
        bev_image = cv2.warpPerspective(original_image, final_H, (x_max - x_min, y_max - y_min), flags=cv2.INTER_CUBIC)

        # --- 인공 경계선 제거 로직 ---
        gray_for_mask = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_for_mask, 1, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        
        # --- 주차 공간 탐지 로직 ---
        gray_bev = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        blurred_bev = cv2.GaussianBlur(gray_bev, (5, 5), 0)
        edges = cv2.Canny(blurred_bev, 60, 120)
        final_edges = cv2.bitwise_and(edges, edges, mask=eroded_mask)

        # cv2.imshow("1. Original Edges", edges)
        # cv2.imshow("2. Eroded Mask", eroded_mask)
        # cv2.imshow("3. Final Edges (After Masking)", final_edges)

        contours, _ = cv2.findContours(final_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_parking_slots = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 3000 or area > 7000: continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) != 4: continue
            (x, y, w, h) = cv2.boundingRect(approx)
            if h == 0: continue
            aspect_ratio = float(w) / h
            if aspect_ratio < 1.0 or aspect_ratio > 1.5: continue


            rospy.loginfo(f"Detected contour with area: {area}")
            rospy.loginfo(f"aspect_ratio: {aspect_ratio:.2f}")
            detected_parking_slots.append(approx)

        # --- 목표 슬롯 선택 및 경로 계산 ---
        target_slot_center = None
        max_cY = 0
        if detected_parking_slots:
            for slot in detected_parking_slots:
                M = cv2.moments(slot)
                if M["m00"] == 0: continue
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if cY > max_cY:
                    max_cY = cY
                    target_slot_center = (cX, cY)

        

        cv2.drawContours(bev_image, detected_parking_slots, -1, (0, 255, 0), 3)
        
        if target_slot_center:
            end_point_px = target_slot_center
            img_h, img_w = bev_image.shape[:2]
            center_line_x = img_w // 2
            angle_degrees = 150.0
            angle_rad = np.deg2rad(angle_degrees)
            slope = np.tan(angle_rad)
            y_on_center_line = end_point_px[1] + slope * (center_line_x - end_point_px[0])
            start_point_px = (center_line_x, int(round(y_on_center_line)))
            
            cv2.line(bev_image, start_point_px, end_point_px, (255, 0, 0), 2)
            cv2.circle(bev_image, start_point_px, 7, (0, 255, 255), -1) # 노란색 시작점
            cv2.putText(bev_image, "Start", (start_point_px[0] + 10, start_point_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(bev_image, end_point_px, 7, (255, 0, 255), -1) # 자주색 끝점
            cv2.putText(bev_image, "End", (end_point_px[0] + 10, end_point_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # === [핵심] === 픽셀 경로를 미터 단위 로컬 경로로 변환하여 발행
            start_point_meter = self.convert_pixel_to_local(start_point_px, bev_image.shape)
            end_point_meter = self.convert_pixel_to_local(end_point_px, bev_image.shape)

            _, start_delta_px, start_meter = self.get_all_coordinates(start_point_px, bev_image.shape)
            _, end_delta_px, end_meter = self.get_all_coordinates(end_point_px, bev_image.shape)

            rospy.loginfo("--- Path Points Information ---")
            rospy.loginfo(f"[START] Pixel Coords: {start_point_px}")
            rospy.loginfo(f"[START] Delta Pixels (from center): {start_delta_px}")
            rospy.loginfo(f"[START] Vehicle Coords (meters): (x={start_meter[0]:.2f}, y={start_meter[1]:.2f})")
            rospy.loginfo("-----------------------------")
            rospy.loginfo(f"[END]   Pixel Coords: {end_point_px}")
            rospy.loginfo(f"[END]   Delta Pixels (from center): {end_delta_px}")
            rospy.loginfo(f"[END]   Vehicle Coords (meters): (x={end_meter[0]:.2f}, y={end_meter[1]:.2f})")
            rospy.loginfo("=============================")
            
            path_meter_msg = Path()
            path_meter_msg.header.stamp = rospy.Time.now()
            path_meter_msg.header.frame_id = "vehicle_local_frame" # 미터 단위 로컬 좌표계임을 명시

            start_pose = PoseStamped()
            start_pose.pose.position.x = start_point_meter[0]
            start_pose.pose.position.y = start_point_meter[1]
            path_meter_msg.poses.append(start_pose)

            end_pose = PoseStamped()
            end_pose.pose.position.x = end_point_meter[0]
            end_pose.pose.position.y = end_point_meter[1]
            path_meter_msg.poses.append(end_pose)

            self.path_meter_pub.publish(path_meter_msg)
            # ==========================================================

        # --- 최종 시각화 이미지 발행 및 화면 표시 ---
        try:
            ros_bev_image = self.bridge.cv2_to_imgmsg(bev_image, "bgr8")
            self.image_pub.publish(ros_bev_image)
        except Exception as e:
            rospy.logerr(f"Image publishing error: {e}")

        cv2.imshow("BEV Parking Detection", bev_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        node = BEVParkingDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass