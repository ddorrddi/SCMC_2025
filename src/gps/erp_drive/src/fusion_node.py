#!/usr/bin/env python3

### TODO : 전체적으로 fusion해서 delivery mission 인지로직은 대부분 이 노드에서 해결
### TODO : delivery_planner는 여기서 발행한 토픽들로 플래그들 세워서 말그대로 플래닝만.
### 퍼블리셔 최종 예측 : 표지판의 좌표와 번호.

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2, PointCloud
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from fusion_module import Fusion
from lidar_object_detection.msg import PointInfo
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

        self.fusion = Fusion(Fx, Fy, Cx, Cy, R, T, img_width, img_height)
        self.cv_image = None
        self.yolo_bbox = None
        self.lidar_bbox = None

        # 이미지 퍼블리셔 (RViz에서 시각화할 토픽)
        self.image_pub = rospy.Publisher('/fusion/image', Image, queue_size=1)
        self.fusion_pub = rospy.Publisher('/fusion/result', IoUFusionMsg, queue_size=1)

        # 서브스크라이버
        ### TODO : 인지팀에서 토픽/메시지 수정해서 줄때까지 스탑.
        self.yolo_sub = Subscriber('/yolo/bbox', PointCloud) # yolo bbox 점 4개 좌표 (이미지 상) (4,2) 와 타겟넘버가 왔으면 좋겠다
        self.lidar_sub = Subscriber('/bbox_point_info_static', PointInfo) # 클러스터 박스들 (bboxCounts, xMini[], yMini[], zMini[], xMaxi[], yMaxi[], zMaxi[])
        self.cam_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback)

        # 근사 시간 동기화
        ats = ApproximateTimeSynchronizer([self.yolo_sub, self.lidar_sub], queue_size=10, slop=0.1)
        ats.registerCallback(self.bbox_callback)

    def bbox_callback(self, yolo_msg, lidar_msg): # TODO : !!인지팀에서 토픽/메시지 픽스나기 전이라서 틀만 짜둠. 픽스나면 그때 세부조정들어가야됨!! (포인트들 읽는방식이나 bbox만드는 방식이나 등등..)
        # 1. YOLO bbox 먼저 생성
        yolo_pts = np.array([[p.x, p.y] for p in yolo_msg.points])
        yolo_bbox = self.fusion.points2bbox(yolo_pts)
        self.yolo_bbox = yolo_bbox.reshape((-1,1,2)).astype(np.int32) # cv2 기능 쓰기위해서 (N, 1, 2)형태로 변환

        # 2. 여러개의 클러스터 박스중 가장 IoU값이 높은 하나를 정하기 위한 변수
        best_iou = 0.
        x = y = z = 0.

        # 3. 각 라이다 클러스터박스들로 bbox 만들고 IoU돌리면서 best하나 고름
        for i in range(lidar_msg.bboxCounts):
            lidar_pts = np.array(list(pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)))
            lidar_pts2d = self.fusion.rid2img(lidar_pts)
            lidar_bbox = self.fusion.points2bbox(lidar_pts2d).reshape((-1,1,2)).astype(np.int32) # OpenCV용 형태

            temp_iou = self.fusion.bbox_iou(self.yolo_bbox, lidar_bbox)
            if best_iou < temp_iou:
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


    def img_callback(self,msg):
        # 카메라 이미지 → OpenCV 형식
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.cv_image = img
        except Exception as e:
            rospy.logwarn(f"CV bridge error: {e}")
            return
        
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

        # 퍼블리시할 메시지로 변환
        fusion_msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding='bgr8')
        fusion_msg.header = msg.header  # 시간 동기화용

        self.image_pub.publish(fusion_msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = FusionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
