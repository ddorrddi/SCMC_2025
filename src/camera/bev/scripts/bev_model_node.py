#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import os

class BEVParkingBoxDetector:
    def __init__(self):
        rospy.init_node('bev_parking_box_detector_node', anonymous=True)

        self.bridge = CvBridge()

        # Homography 행렬
        self.H = np.array([
            [-9.49144753e-02, -1.37770938e+00, 3.60924395e+02],
            [ 1.45962627e-02, -9.91594887e-01,  2.69889792e+02],
            [ 7.34354190e-05, -4.27074219e-03,  1.00000000e+00]
        ])
        
        # 실제 스케일 값 (m/pixel)
        self.METERS_PER_PIXEL = 0.053

        # YOLO 모델 로드
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', '0831Best.pt')
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except Exception as e:
            rospy.logerr(f"Failed to load YOLOv8 model: {e}")
            return

        # Publisher 설정
        self.image_pub = rospy.Publisher("/bev_parking_image", Image, queue_size=1)
        self.path_meter_pub = rospy.Publisher('/parking_path_meter', Path, queue_size=1)
        
        # Subscriber 설정
        self.image_sub = rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback, queue_size=1)


    def convert_pixel_to_local(self, point_px, image_shape):
        """픽셀을 차량 로컬 좌표(m)로 변환"""
        img_h, img_w = image_shape[:2]
        image_center_px = (img_w // 2, img_h // 2)
        
        delta_px_x = point_px[0] - image_center_px[0]
        delta_px_y = point_px[1] - image_center_px[1]

        real_x = -delta_px_y * self.METERS_PER_PIXEL # 전방: +X
        real_y = -delta_px_x * self.METERS_PER_PIXEL # 좌측: +Y
        
        return (real_x, real_y)
    
    def calculate_box_center(self, box_points):
        """바운딩 박스 꼭짓점들의 중심점 계산"""
        center_x = sum(point[0] for point in box_points) / len(box_points)
        center_y = sum(point[1] for point in box_points) / len(box_points)
        return (int(center_x), int(center_y))

    def detect_parking_boxes(self, bev_image):
        """YOLOv8 모델을 사용하여 바운딩 박스 형태의 주차 공간 탐지"""
        try:
            results = self.model(bev_image, conf=0.5, verbose=False)
            detected_parking_boxes = []
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        if conf > 0.5:
                            box_points = [
                                (int(x1), int(y1)), (int(x2), int(y1)),
                                (int(x2), int(y2)), (int(x1), int(y2))
                            ]
                            center_point = self.calculate_box_center(box_points)
                            box_info = {
                                'box_points': box_points,
                                'center': center_point,
                                'confidence': float(conf),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1)
                            }
                            detected_parking_boxes.append(box_info)
            
            return detected_parking_boxes
            
        except Exception as e:
            rospy.logerr(f"YOLOv8 box detection error: {e}")
            return []

    def image_callback(self, ros_image_compressed):
        try:
            np_arr = np.frombuffer(ros_image_compressed.data, np.uint8)
            original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if original_image is None: 
                rospy.logerr("Failed to decode image")
                return
            
            original_image[0:270, :] = 0
        except Exception as e:
            rospy.logerr(f"Image decoding error: {e}")
            return
            
        # --- BEV 변환 ---
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

        cv2.imshow("BEV Image", bev_image)

        # --- YOLO 모델을 사용한 바운딩 박스 주차 공간 탐지 ---
        detected_parking_boxes = self.detect_parking_boxes(bev_image)
        
        # --- 목표 슬롯 선택을 위한 조건 필터링 ---
        
        # 조건 1: 정사각형에 가까운 박스만 필터링
        square_like_boxes = []
        for box_data in detected_parking_boxes:
            w = box_data['width']
            h = box_data['height']
            if w == 0 or h == 0: continue # 너비나 높이가 0이면 스킵

            aspect_ratio = min(w, h) / max(w, h) # 짧은 변 / 긴 변
            if aspect_ratio >= 0.9: # 비율이 0.8 이상인 경우만 추가
                square_like_boxes.append(box_data)

        # --- 필터링된 박스들 시각화 및 목표 선택 ---
        target_slot_center = None
        max_cY = 0
        
        # 필터링을 통과한 모든 유효한 박스를 시각화
        for box_data in square_like_boxes:
            box_points = box_data['box_points']
            center = box_data['center']
            conf = box_data['confidence']
            
            # 박스 그리기 (초록색)
            pts = np.array(box_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(bev_image, [pts], True, (0, 255, 0), 3)
            overlay = bev_image.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            bev_image = cv2.addWeighted(bev_image, 0.7, overlay, 0.3, 0)
            
            # 중심점 표시 (빨간색)
            cv2.circle(bev_image, center, 8, (0, 0, 255), -1)
            
            # confidence 표시
            cv2.putText(bev_image, f"{conf:.2f}", (center[0] - 20, center[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 조건 2: 필터링된 박스들 중 가장 아래에 있는 것을 목표로 선택
            if center[1] > max_cY:
                max_cY = center[1]
                target_slot_center = center

        # --- 경로 생성 및 발행 ---
        if target_slot_center:
            # (경로 생성 및 Publish 로직은 동일)
            end_point_px = target_slot_center
            img_w = bev_image.shape[1]
            center_line_x = img_w // 2
            angle_degrees = 150.0
            angle_rad = np.deg2rad(angle_degrees)
            slope = np.tan(angle_rad)
            y_on_center_line = end_point_px[1] + slope * (center_line_x - end_point_px[0])
            start_point_px = (center_line_x, int(round(y_on_center_line)))
            
            cv2.line(bev_image, start_point_px, end_point_px, (255, 0, 0), 3)
            cv2.circle(bev_image, start_point_px, 10, (0, 255, 255), -1)
            cv2.putText(bev_image, "Start", (start_point_px[0] + 15, start_point_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.circle(bev_image, end_point_px, 10, (255, 0, 255), -1)
            cv2.putText(bev_image, "End", (end_point_px[0] + 15, end_point_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            start_point_meter = self.convert_pixel_to_local(start_point_px, bev_image.shape)
            end_point_meter = self.convert_pixel_to_local(end_point_px, bev_image.shape)
            
            path_meter_msg = Path()
            path_meter_msg.header.stamp = rospy.Time.now()
            path_meter_msg.header.frame_id = "vehicle_local_frame"

            start_pose = PoseStamped()
            start_pose.pose.position.x = start_point_meter[0]
            start_pose.pose.position.y = start_point_meter[1]
            path_meter_msg.poses.append(start_pose)

            end_pose = PoseStamped()
            end_pose.pose.position.x = end_point_meter[0]
            end_pose.pose.position.y = end_point_meter[1]
            path_meter_msg.poses.append(end_pose)

            self.path_meter_pub.publish(path_meter_msg)

        # --- 최종 시각화 ---
        try:
            ros_bev_image = self.bridge.cv2_to_imgmsg(bev_image, "bgr8")
            self.image_pub.publish(ros_bev_image)
            cv2.imshow("BEV Parking Box Detection", bev_image)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Image publishing or display error: {e}")

if __name__ == '__main__':
    try:
        node = BEVParkingBoxDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass