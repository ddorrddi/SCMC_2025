#!/usr/bin/env python3

### TODO : 전체적으로 fusion해서 delivery mission 인지로직은 대부분 이 노드에서 해결
### TODO : delivery_planner는 여기서 발행한 토픽들로 플래그들 세워서 말그대로 플래닝만.
### 퍼블리셔 최종 예측 : 표지판의 좌표와 번호.

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from fusion_module import Fusion
from lidar_object_detection.msg import PointInfo
from delivery.msg import DetectYolo
from erp_drive.msg import IoUFusionMsg

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
        self.img_width = img_width
        self.img_height = img_height

        self.fusion = Fusion(Fx, Fy, Cx, Cy, R, T, img_width, img_height)
        self.cv_image = None
        self.yolo_bbox = None
        self.lidar_bbox = None
        self.lidar_pts2d = None

        self.cloud_stride = 8
        self.max_projected_points = 2000

        # 이미지 퍼블리셔 (RViz에서 시각화할 토픽)
        self.image_pub = rospy.Publisher('/fusion/image', Image, queue_size=1)
        self.cam_info_pub = rospy.Publisher('/fusion/camera_info', CameraInfo, queue_size=1)
        self.fusion_pub = rospy.Publisher('/fusion/result', IoUFusionMsg, queue_size=1)

        # 서브스크라이버
        ### TODO : 인지팀에서 토픽/메시지 수정해서 줄때까지 스탑.
        self.yolo_sub = Subscriber('/delivery_sign_info', DetectYolo) # 욜로 bbox : class_name, center_x, center_y, width, height
        self.lidar_sub = Subscriber('/bbox_point_info_static', PointInfo) # 클러스터 박스들 (bboxCounts, xMini[], yMini[], zMini[], xMaxi[], yMaxi[], zMaxi[])
        self.img_sub = Subscriber('/usb_cam_right/image_raw', Image)
        self.info_sub = Subscriber('/usb_cam_right/camera_info', CameraInfo)
        self.cloud_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.cloud_callback, queue_size=1)

        # 근사 시간 동기화
        ats = ApproximateTimeSynchronizer([self.yolo_sub, self.lidar_sub], queue_size=10, slop=0.1, allow_headerless=True)
        ats.registerCallback(self.bbox_callback)

        ats_cam = ApproximateTimeSynchronizer([self.img_sub, self.info_sub], queue_size=10, slop=0.1)
        ats_cam.registerCallback(self.img_callback)


    def bbox_callback(self, yolo_msg, lidar_msg): # TODO : !!인지팀에서 토픽/메시지 픽스나기 전이라서 틀만 짜둠. 픽스나면 그때 세부조정들어가야됨!! (포인트들 읽는방식이나 bbox만드는 방식이나 등등..)
        # 1. YOLO bbox 먼저 생성
        x_half = yolo_msg.width // 2
        y_half = yolo_msg.height // 2
        points = [(yolo_msg.center_x + x_half, yolo_msg.center_y + y_half), 
                  (yolo_msg.center_x - x_half, yolo_msg.center_y + y_half), 
                  (yolo_msg.center_x - x_half, yolo_msg.center_y - y_half), 
                  (yolo_msg.center_x + x_half, yolo_msg.center_y - y_half)]
        yolo_pts = np.array([[p[0], p[1]] for p in points])
        yolo_bbox = self.fusion.points2bbox(yolo_pts)
        self.yolo_bbox = yolo_bbox.reshape((-1,1,2)).astype(np.int32) # cv2 기능 쓰기위해서 (N, 1, 2)형태로 변환

        # 2. 여러개의 클러스터 박스중 가장 IoU값이 높은 하나를 정하기 위한 변수
        best_iou = 0.
        x = y = z = 0.

        # 3. 각 라이다 클러스터박스들로 bbox 만들고 IoU돌리면서 best하나 고름
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

            temp_iou = self.fusion.bbox_iou(self.yolo_bbox.squeeze(), lidar_bbox.squeeze())
            if best_iou <= temp_iou:
                best_iou = temp_iou
                x = (lidar_msg.xMini[i] + lidar_msg.xMaxi[i]) / 2
                y = (lidar_msg.yMini[i] + lidar_msg.yMaxi[i]) / 2
                z = (lidar_msg.zMini[i] + lidar_msg.zMaxi[i]) / 2
                self.lidar_bbox = lidar_bbox
        
        # 4. 
        if best_iou > 0:
            msg = IoUFusionMsg()
            msg.dest_num = yolo_msg.num
            msg.score = best_iou
            msg.x = x
            msg.y = y
            msg.z = z
            self.fusion_pub.publish(msg)


    def cloud_callback(self, cloud_msg: PointCloud2):
        """
        /velodyne_points 수신 → 일부 샘플링 → 카메라로 투영 → 이미지 좌표 배열을 self.lidar_pts2d에 저장
        실제 그리기는 img_callback에서 수행 (카메라 타이밍에 맞춤)
        """
        try:
            # PointCloud2 → (x,y,z) 제너레이터
            gen = pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)

            # 부하를 줄이기 위한 stride 샘플링 및 상한
            sampled = []
            for idx, (x, y, z) in enumerate(gen):
                if idx % self.cloud_stride == 0:
                    sampled.append([x, y, z])
                    if len(sampled) >= self.max_projected_points * 3:  # 여유 샘플(투영 후 화면 밖 제외 대비)
                        break

            if not sampled:
                self.lidar_pts2d = None
                return

            lidar_pts = np.asarray(sampled, dtype=np.float32)  # (N,3)

            # LiDAR → 이미지 좌표로 투영
            pts2d = self.fusion.rid2img(lidar_pts)  # (N,2) expected
            if pts2d is None or len(pts2d) == 0:
                self.lidar_pts2d = None
                return

            # 이미지 경계 내부만 필터링
            u = pts2d[:, 0]
            v = pts2d[:, 1]
            mask = (u >= 0) & (u < self.img_width) & (v >= 0) & (v < self.img_height)
            pts2d = pts2d[mask]
            lidar_pts = lidar_pts[mask]

            if pts2d.shape[0] == 0:
                self.lidar_pts2d = None
                return
            
            # 거리계산
            dist = np.linalg.norm(lidar_pts, axis=1)

            # 너무 많으면 랜덤 샘플 (그리기 성능 보호)
            if pts2d.shape[0] > self.max_projected_points:
                idx = np.random.choice(pts2d.shape[0], self.max_projected_points, replace=False)
                pts2d = pts2d[idx]
                dist = dist[idx]

            # (u,v,d) 형태로 저장
            self.lidar_pts2d = np.hstack([pts2d.astype(np.int32), dist.reshape(-1, 1)])

        except Exception as e:
            rospy.logwarn(f"[cloud_callback] error: {e}")
            self.lidar_pts2d = None


    def img_callback(self,img_msg, info_msg):
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
            
        if self.lidar_bbox is not None:
            pts = self.lidar_bbox.reshape(-1, 2)
            for (u, v) in pts:
                cv2.circle(self.cv_image, (int(u), int(v)), 4, (0, 255, 255), -1)  # 노랑 점
            cv2.polylines(self.cv_image, [self.lidar_bbox], isClosed=True, color=(0, 255, 255), thickness=2) # 노란색 박스
        
        # LiDAR 포인트 그리기 (거리 기반 색상)
        if self.lidar_pts2d is not None and self.lidar_pts2d.size > 0:
            for (u, v, d) in self.lidar_pts2d:
                if d > 40:         # 40m 이상 → 회색
                    color = (128, 128, 128)
                elif d < 5:        # 5m 이내 → 빨강
                    color = (0, 0, 255)
                elif d < 20:       # 5~20m → 빨강→노랑 보간
                    ratio = (d - 5) / 15.0
                    color = (0, int(255 * ratio), 255)  # BGR
                else:              # 20~40m → 노랑→초록 보간
                    ratio = (d - 20) / 20.0
                    color = (0, 255, int(255 * (1 - ratio)))

                cv2.circle(self.cv_image, (int(u), int(v)), 2, color, -1)

        # 퍼블리시할 메시지로 변환
        fusion_msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
        fusion_msg.header = img_msg.header  # 시간 동기화용

        info_msg.header = img_msg.header

        self.image_pub.publish(fusion_msg)
        self.cam_info_pub.publish(info_msg)


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = FusionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
