#!/usr/bin/env python3

### TODO : 전체적으로 fusion해서 delivery mission 인지로직은 대부분 이 노드에서 해결
### TODO : delivery_planner는 여기서 발행한 토픽들로 플래그들 세워서 말그대로 플래닝만.
### 퍼블리셔 최종 예측 : 표지판의 좌표와 번호.

import rospy
import numpy as np
import cv2

from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from fusion_module import Fusion
from lidar_object_detection.msg import PointInfo
from delivery.msg import DetectYolo
from erp_drive.msg import IoUFusionMsg

class FusionNode:
    def __init__(self):
        rospy.init_node('fusion_node')
        self.bridge = CvBridge()

        # 내부 파라미터
        Fx, Fy = 511, 513
        Cx, Cy = 324, 241

        # 외부 파라미터
        x, y, z = 0.0, 0.0, 0.0        # m 단위
        roll, pitch, yaw = 0.0, 0.0, 0.0  # rad 단위
        R, T = Fusion.make_extrinsics(x, y, z, roll, pitch, yaw)

        # 카메라 해상도
        img_width = 640
        img_height = 480
        self.img_width = img_width
        self.img_height = img_height

        # Fusion 객체
        self.fusion = Fusion(Fx, Fy, Cx, Cy, R, T, img_width, img_height)

        self.cv_image = None
        self.yolo_bbox = None
        self.lidar_bboxes = []
        self.lidar_pts2d = None

        # 포인트클라우드 시각화 파라미터
        self.cloud_stride = 1
        self.max_projected_points = 10000

        # 이미지 퍼블리셔 (RViz에서 시각화할 토픽)
        self.image_pub = rospy.Publisher('/fusion/image', Image, queue_size=1)
        self.fusion_pub = rospy.Publisher('/fusion/result', IoUFusionMsg, queue_size=1)

        # 서브스크라이버
        self.yolo_sub = rospy.Subscriber('/delivery_sign_info', DetectYolo, self.yolo_bbox_callback, queue_size=1) # 욜로 bbox : class_name, center_x, center_y, width, height
        self.lidar_sub = rospy.Subscriber('/bbox_point_info_static', PointInfo, self.lidar_bbox_callback, queue_size=1) # 클러스터 박스들 (bboxCounts, xMini[], yMini[], zMini[], xMaxi[], yMaxi[], zMaxi[])
        self.img_sub = rospy.Subscriber('/usb_cam_right/image_raw', Image, self.img_callback, queue_size=1)
        self.cloud_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.cloud_callback, queue_size=1)


    def yolo_bbox_callback(self, yolo_msg):
        # YOLO bbox 저장
        x_half = yolo_msg.width // 2
        y_half = yolo_msg.height // 2
        points = [(yolo_msg.center_x + x_half, yolo_msg.center_y + y_half), 
                  (yolo_msg.center_x - x_half, yolo_msg.center_y + y_half), 
                  (yolo_msg.center_x - x_half, yolo_msg.center_y - y_half), 
                  (yolo_msg.center_x + x_half, yolo_msg.center_y - y_half)]
        yolo_pts = np.array([[p[0], p[1]] for p in points])
        yolo_bbox = self.fusion.points2bbox(yolo_pts)
        self.yolo_bbox = yolo_bbox.reshape((-1,1,2)).astype(np.int32) # cv2 기능 쓰기위해서 (N, 1, 2)형태로 변환


    def lidar_bbox_callback(self, lidar_msg):
        # 라이다 bbox 리스트 저장
        for i in range(lidar_msg.bboxCounts):
            x_min, y_min, z_min = lidar_msg.xMini[i], lidar_msg.yMini[i], lidar_msg.zMini[i]
            x_max, y_max, z_max = lidar_msg.xMaxi[i], lidar_msg.yMaxi[i], lidar_msg.zMaxi[i]

            corners_3d = [
                (x_min, y_min, z_min),
                (x_min, y_min, z_max),
                (x_min, y_max, z_min),
                (x_min, y_max, z_max),
                (x_max, y_min, z_min),
                (x_max, y_min, z_max),
                (x_max, y_max, z_min),
                (x_max, y_max, z_max),
            ]

            lidar_pts = np.array(corners_3d, dtype=np.float32)
            lidar_pts2d = self.fusion.rid2img(lidar_pts)
            if lidar_pts2d is None or len(lidar_pts2d) == 0:continue

            lidar_bbox = self.fusion.points2bbox(lidar_pts2d).reshape((-1,1,2)).astype(np.int32) # OpenCV용 형태
            self.lidar_bboxes.append(lidar_bbox)


    def cloud_callback(self, cloud_msg: PointCloud2):
        try:
            gen = pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)

            # stride 기반 샘플링 (속도 제어)
            sampled = []
            for idx, (x, y, z) in enumerate(gen):
                if idx % self.cloud_stride == 0:
                    sampled.append([x, y, z])
                    if len(sampled) >= self.max_projected_points * 3:
                        break

            if not sampled:
                self.lidar_pts2d = None
                return

            lidar_pts = np.asarray(sampled, dtype=np.float32)  # [N,3]

            # ★ 유효 인덱스 함께 받기 (Z>0 필터 반영)
            pts2d, valid_idx = self.fusion.rid2img(lidar_pts, return_mask=True)  # pts2d:[M,2], valid_idx:[M]
            if pts2d is None or len(pts2d) == 0:
                self.lidar_pts2d = None
                return

            # 동일한 점들만 남기도록 3D도 정렬
            lidar_pts = lidar_pts[valid_idx]  # [M,3] ← pts2d와 길이 일치 보장

            # 이미지 경계 내부만 필터링
            u = pts2d[:, 0]
            v = pts2d[:, 1]
            inb = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height)

            if not np.any(inb):
                self.lidar_pts2d = None
                return

            pts2d = pts2d[inb]         # [K,2]
            lidar_pts = lidar_pts[inb] # [K,3]

            # 거리 계산
            dist = np.linalg.norm(lidar_pts, axis=1)  # [K]

            # 너무 많으면 랜덤 샘플링
            if pts2d.shape[0] > self.max_projected_points:
                sel = np.random.choice(pts2d.shape[0], self.max_projected_points, replace=False)
                pts2d = pts2d[sel]
                dist  = dist[sel]

            # (u,v,d) 저장 (그리기는 img_callback에서)
            uv = pts2d.astype(np.int32)
            self.lidar_pts2d = np.column_stack([uv, dist])

        except Exception as e:
            rospy.logwarn(f"[cloud_callback] error: {e}")
            self.lidar_pts2d = None


    def img_callback(self, img_msg):
        # 카메라 이미지 → OpenCV 형식
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            self.cv_image = img
        except Exception as e:
            rospy.logwarn(f"CV bridge error: {e}")
            return
        
        # BBOX 그리기
        if self.yolo_bbox is not None:
            # (4,1,2) → (4,2) 형태로 바꿔서 순회
            pts = self.yolo_bbox.reshape(-1, 2)
            for (u, v) in pts:
                cv2.circle(self.cv_image, (int(u), int(v)), 4, (0, 0, 255), -1)  # 빨강 점
            cv2.polylines(self.cv_image, [self.yolo_bbox], isClosed=True, color=(0, 0, 255), thickness=2) # 빨간색 박스
            
        if self.lidar_bbox:
            for lidar_bbox in self.lidar_bboxes:
                pts = lidar_bbox.reshape(-1, 2)
                for (u, v) in pts:
                    cv2.circle(self.cv_image, (int(u), int(v)), 4, (0, 255, 255), -1)  # 노랑 점
                cv2.polylines(self.cv_image, [lidar_bbox], isClosed=True, color=(0, 255, 255), thickness=2) # 노란색 박스
        
        # LiDAR 포인트 그리기 (거리 기반 색상)
        if self.lidar_pts2d is not None and self.lidar_pts2d.size > 0:
            for (u, v, d) in self.lidar_pts2d:
                if d > 5:         # 5m 이상 → 회색
                    color = (128, 128, 128)
                elif d < 1:        # 1m 이내 → 빨강
                    color = (0, 0, 255)
                elif d < 3:       # 1~3m → 빨강→노랑 보간
                    ratio = (d - 1) / 2.0
                    color = (0, int(255 * ratio), 255)  # BGR
                else:              # 3~5m → 노랑→초록 보간
                    ratio = (d - 3) / 2.0
                    color = (0, 255, int(255 * (1 - ratio)))

                cv2.circle(self.cv_image, (int(u), int(v)), 2, color, -1)

        # 퍼블리시할 메시지로 변환
        fusion_msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
        fusion_msg.header = img_msg.header  # 시간 동기화용

        self.image_pub.publish(fusion_msg)


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = FusionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
