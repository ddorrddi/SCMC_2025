#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2, PointCloud
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from fusion_module import Fusion

class FusionNode:
    def __init__(self):
        rospy.init_node('fusion_node')
        self.bridge = CvBridge()

        # 카메라 파라미터값 (일단 임시값)
        Fx, Fy = 511, 513
        Cx, Cy = 324, 241
        R = [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0]
        ]
        T = [
            [0.0],
            [0.0],
            [0.0]
        ]
        img_width = 640
        img_height = 480

        self.fusion = Fusion(Fx, Fy, Cx, Cy, R, T, img_width, img_height)
        self.yolo_bbox = None
        self.lidar_bbox = None

        # 이미지 퍼블리셔 (RViz에서 시각화할 토픽)
        self.image_pub = rospy.Publisher('/fusion/image', Image, queue_size=1)

        # 토픽 구독자 설정 (image + lidar)
        self.image_sub = Subscriber('/usb_cam/image_raw', Image)
        self.lidar_sub = Subscriber('/velodyne_points', PointCloud2)

        # 일단 주석화
        self.yolo_bbox_sub = rospy.Subscriber('/yolo/bbox', PointCloud, self.yolo_callback) # yolo bbox 점 4개 좌표 (이미지 상) (4,2)
        self.lidar_bbox_sub = rospy.Subscriber('/lidar/bbox_3d', PointCloud2, self.lidar_callback) # cluster box 점 8개 좌표 (라이다 상) (8,3)


        # 근사 시간 동기화
        ats = ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], queue_size=10, slop=0.1)
        ats.registerCallback(self.callback)

    def callback(self, image_msg, cloud_msg):
        # 카메라 이미지 → OpenCV 형식
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"CV bridge error: {e}")
            return
        
        if self.yolo_bbox is not None:
            # (4,1,2) → (4,2) 형태로 바꿔서 순회
            pts = self.yolo_bbox.reshape(-1, 2)
            for (u, v) in pts:
                cv2.circle(cv_image, (int(u), int(v)), 4, (0, 0, 255), -1)  # 빨강 점
            cv2.polylines(cv_image, [self.yolo_bbox], isClosed=True, color=(0, 0, 255), thickness=2) # 빨간색 박스
            
        if self.lidar_bbox is not None:
            pts = self.lidar_bbox.reshape(-1, 2)
            for (u, v) in pts:
                cv2.circle(cv_image, (int(u), int(v)), 4, (0, 255, 255), -1)  # 노랑 점
            cv2.polylines(cv_image, [self.lidar_bbox], isClosed=True, color=(0, 255, 255), thickness=2) # 노란색 박스

        # 라이다 포인트 읽기
        points = []
        for p in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([p[0], p[1], p[2]])
        points_np = np.array(points)

        # 라이다 포인트를 이미지로 투영
        try:
            points_2d = self.fusion.rid2img(points_np)
        except Exception as e:
            rospy.logwarn(f"Fusion error: {e}")
            return

        # 이미지 위에 포인트 시각화
        for u, v in points_2d.astype(int):
            if 0 <= u < self.fusion.img_width and 0 <= v < self.fusion.img_height:
                cv2.circle(cv_image, (u, v), 2, (0, 255, 0), -1)  # 초록색 점

        # 퍼블리시할 메시지로 변환
        fusion_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        fusion_msg.header = image_msg.header  # 시간 동기화용

        self.image_pub.publish(fusion_msg)

    def yolo_callback(self, msg):
        try:
            points_np = np.array([[p.x, p.y] for p in msg.points])
        except Exception as e:
            rospy.logwarn(f"YOLO bbox parsing failed: {e}")
            return
        bbox = self.fusion.points2bbox(points_np)
        self.yolo_bbox = bbox.reshape((-1,1,2)).astype(np.int32) # cv2 기능 쓰기위해서 (N, 1, 2)형태로 변환

    def lidar_callback(self, msg):
        try:
            # PointCloud2 → 리스트 of (x,y,z)
            points = list(pc2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            ))
            points_np = np.array(points)  # shape (N,3)

            # (옵션) BBox 전용 8점만 보내는 메시지를 쓰셨다면
            # points_np = points_np[:8]  # 예시로 처음 8개만 가져오기
        except Exception as e:
            rospy.logwarn(f"Cluster box msg not convertible to ndarray: {e}")
            return

        # 라이다 포인트를 이미지로 투영
        try:
            points_2d = self.fusion.rid2img(points_np)
        except Exception as e:
            rospy.logwarn(f"Fusion projection error: {e}")
            return

        # 이미지 위에 Bbox 시각화
        try:
            bbox = self.fusion.points2bbox(points_2d)
            self.lidar_bbox = bbox.reshape((-1, 1, 2)).astype(np.int32)  # OpenCV용 형태
        except Exception as e:
            rospy.logwarn(f"Failed to compute lidar bbox: {e}")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = FusionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
