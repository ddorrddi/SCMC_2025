#!/usr/bin/env python3

### TODO : 전체적으로 fusion해서 delivery mission 인지로직은 대부분 이 노드에서 해결
### TODO : delivery_planner는 여기서 발행한 토픽들로 플래그들 세워서 말그대로 플래닝만.
### 퍼블리셔 최종 예측 : 표지판의 좌표와 번호.

import rospy
import numpy as np
import cv2

from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage

from geometry_msgs.msg import PointStamped

from fusion_module import Fusion
from delivery.msg import DetectYolo

from dynamic_reconfigure.server import Server
from erp_drive.cfg import FusionExtrinsicsConfig

class FusionNode:
    def __init__(self):
        rospy.init_node('fusion_node')
        self.bridge = CvBridge()

        # 내부 파라미터
        Fx, Fy = 511, 513
        Cx, Cy = 324, 241

        # 외부 파라미터
        self.x, self.y, self.z = 0.0, 0.0, 0.0        # m 단위
        self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0  # rad 단위

        # 카메라 해상도
        self.img_width = 640
        self.img_height = 480

        # 초기 R, T
        self.R, self.T = Fusion.make_extrinsics(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)

        # Fusion 객체
        self.fusion = Fusion(Fx, Fy, Cx, Cy, self.R, self.T, self.img_width, self.img_height)

        # 이미지와 포인트 초기화
        self.cv_image = None
        self.lidar_pts2d = None

        # === New: YOLO bbox & Cloud 상태 ===
        self.yolo_bbox = None             # (4,1,2) poly for drawing
        self.yolo_xyxy = None             # (x1,y1,x2,y2) for masking
        self.cloud_uv  = None             # (K,2) float32
        self.cloud_xyz = None             # (K,3) float32

        # === New: 정제 파이프라인 파라미터 ===
        self.r_cut = 0.8
        # [0.6~0.95] YOLO bbox 중심에서의 정규 반경 컷(r <= r_cut).
        # ↓(0.6~0.75): 박스 가장자리 노이즈/칼리브 불안정 → 더 중심부만 사용
        # ↑(0.85~0.95): 박스가 타이트하거나 칼리브 정확 → 더 많은 포인트 허용

        self.sigma2d = 0.5
        # [0.3~0.8] bbox 중심 가중치의 분산(가우시안 너비).
        # ↓(0.3~0.45): 중심부에 더 강하게 가중(외곽 영향 줄이기)
        # ↑(0.6~0.8): 중심-외곽 가중 차이를 완화(포인트 적을 때 유리)

        self.k_mad = 2.5
        # [2.0~3.5] 거리(범위) 분포의 MAD 기반 이상치 컷 강도.
        # ↓(~2.0): 먼 배경/엉뚱한 포인트가 섞일 때 더 빡세게 제거
        # ↑(~3.0+): 포인트가 적거나 분포가 넓어 과컷되는 경우 완화

        self.neigh_radius_base = 0.25
        # [0.2~0.4] 근접도(반경-이웃) 코어 선택의 최소 반경 [m].
        # ↓: 근거리에서 서로 다른 물체가 섞일 때 군집을 더 작게
        # ↑: 라이다가 듬성듬성 찍히거나 장거리에서 밀도가 낮을 때 완화

        self.neigh_radius_scale = 0.02
        # [0.01~0.04] 거리 비례 반경 스케일(대략 반경 ≈ base 또는 0.02*중앙거리).
        # ↓: 장거리에서도 군집 반경을 보수적으로(혼입 줄이기)
        # ↑: 장거리에서 포인트가 매우 희박해 군집이 잘 안 잡힐 때

        self.neigh_min_samples = 3
        # [2~5] 코어로 인정할 최소 이웃 수.
        # ↓: 라이다 리턴이 희박(비/안개/장거리)해도 통과가 쉬움
        # ↑: 오검출/잡음이 많을 때 합의 강도를 높여 안정화

        self.prev_center = None
        # 이전 프레임의 추정 3D 중심. None이면 시간 게이팅 비활성.
        # (칼만/EMA 없이도 간단히 프레임 간 일관성 확보용으로 사용)

        self.prev_gate = 1.5
        # [0.8~3.0] 시간 게이팅 반경 [m].
        # ↓(~1.0): 표지판처럼 정적 타깃에서 점프/스위칭 방지(더 엄격)
        # ↑(~2.0+): 고속 주행/프레임 드랍/칼리브 덜맞음으로 중심 이동이 큰 환경

        self.max_range = 10.0
        # [1.0~15.0] 탐지된 포인트 최대 거리 임계값

        # 신뢰도 민감도
        self.tau_z2 = 0.36   # (0.6m)^2
        self.tau_2d = 0.25

        # 포인트클라우드 시각화 파라미터
        self.cloud_stride = 1
        self.max_projected_points = 100000

        # 이미지 퍼블리셔 (RViz에서 시각화할 토픽)
        self.image_pub = rospy.Publisher('/fusion/image', Image, queue_size=1)

        # 목적지 좌표 퍼블리셔
        self.center_pub = rospy.Publisher('/fusion/matched_center', PointStamped, queue_size=1)

        # 서브스크라이버
        self.yolo_sub = rospy.Subscriber('/delivery_sign_info', DetectYolo, self.yolo_bbox_callback, queue_size=1) # 욜로 bbox : class_name, center_x, center_y, width, height
        self.img_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.img_callback, queue_size=1)
        # self.img_sub = rospy.Subscriber('/image_jpeg/compressed', CompressedImage, self.img_callback, queue_size=1) # Morai용 토픽
        self.cloud_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.cloud_callback, queue_size=1)

        # Dynamic Reconfigure 서버 시작
        self.dr_srv = Server(FusionExtrinsicsConfig, self.reconfigure_callback)

    def reconfigure_callback(self, config, level):
        # 동적 재구성 콜백
        self.x = config.x
        self.y = config.y
        self.z = config.z
        self.roll = config.roll_deg * np.pi / 180.0
        self.pitch = config.pitch_deg * np.pi / 180.0
        self.yaw = config.yaw_deg * np.pi / 180.0
        self.cloud_stride = config.cloud_stride
        self.max_projected_points = config.max_projected_points
        self.r_cut = config.r_cut
        self.k_mad = config.k_mad
        self.prev_gate = config.prev_gate
        self.max_range = config.max_range
        
        # Extrinsics 업데이트
        self.R, self.T = Fusion.make_extrinsics(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)
        self.fusion.update_extrinsics(self.R, self.T)

        return config

    def yolo_bbox_callback(self, yolo_msg):
        # 정밀 중심/폭/높이
        cx = float(yolo_msg.center_x)
        cy = float(yolo_msg.center_y)
        hw = float(yolo_msg.width)  * 0.5
        hh = float(yolo_msg.height) * 0.5

        x1, y1 = cx - hw, cy - hh
        x2, y2 = cx + hw, cy + hh
        self.yolo_xyxy = (x1, y1, x2, y2)

        # 그리기용 폴리곤
        yolo_pts = np.array([[x1, y1],[x2, y1],[x2, y2],[x1, y2]], dtype=np.float32)
        yolo_bbox = self.fusion.points2bbox(yolo_pts)
        self.yolo_bbox = yolo_bbox.reshape((-1,1,2)).astype(np.int32)

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
                lidar_pts = lidar_pts[sel]

            # (u,v,d) 저장 (그리기는 img_callback에서)
            uv_i32 = pts2d.astype(np.int32)
            self.lidar_pts2d = np.column_stack([uv_i32, dist])

            # === New: 평균 산출용 원본(동일 인덱스) 보관 ===
            self.cloud_uv  = pts2d.astype(np.float32)   # [K,2]
            self.cloud_xyz = lidar_pts.astype(np.float32)  # [K,3]

        except Exception as e:
            rospy.logwarn(f"[cloud_callback] error: {e}")
            self.lidar_pts2d = None


    def img_callback(self, img_msg):
        # 카메라 이미지 → OpenCV 형식
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            # img = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='bgr8') # Morai용
            self.cv_image = img
        except Exception as e:
            rospy.logwarn(f"CV bridge error: {e}")
            return

        # YOLO 박스 (빨강)
        if self.yolo_bbox is not None:
            # (4,1,2) → (4,2) 형태로 바꿔서 순회
            pts = self.yolo_bbox.reshape(-1, 2)
            for (u, v) in pts:
                cv2.circle(self.cv_image, (int(u), int(v)), 4, (0, 0, 255), -1)  # 빨강 점
            cv2.polylines(self.cv_image, [self.yolo_bbox], isClosed=True, color=(0, 0, 255), thickness=2) # 빨간색 박스

        # LiDAR 포인트 그리기 (거리 기반 색상)
        if self.lidar_pts2d is not None and self.lidar_pts2d.size > 0:
            for (u, v, d) in self.lidar_pts2d:
                if d > 10:         # 10m 이상 → 회색
                    color = (128, 128, 128)
                elif d < 1:        # 1m 이내 → 빨강
                    color = (0, 0, 255)
                elif d < 5:       # 1~5m → 빨강→노랑 보간
                    ratio = (d - 1) / 4
                    color = (0, int(255 * ratio), 255)  # BGR
                else:              # 5~10m → 노랑→초록 보간
                    ratio = (d - 5) / 5.0
                    color = (0, 255, int(255 * (1 - ratio)))

                cv2.circle(self.cv_image, (int(u), int(v)), 2, color, -1)

        if self.yolo_xyxy is not None and self.cloud_uv is not None and self.cloud_xyz is not None:
            p3d, conf, inlier_mask = self.fusion.compute_target_from_bbox(
                self.cloud_uv, self.cloud_xyz, self.yolo_xyxy,
                r_cut=self.r_cut, sigma2d=self.sigma2d, k_mad=self.k_mad,
                neigh_radius_base=self.neigh_radius_base,
                neigh_radius_scale=self.neigh_radius_scale,
                neigh_min_samples=self.neigh_min_samples,
                prev_center=self.prev_center, prev_gate=self.prev_gate,
                tau_z2=self.tau_z2, tau_2d=self.tau_2d,
            )
            if p3d is not None:
                # 퍼블리시
                pt = PointStamped()
                pt.header = img_msg.header
                pt.header.frame_id = "velodyne"
                pt.point.x, pt.point.y, pt.point.z = float(p3d[0]), float(p3d[1]), float(p3d[2])
                self.center_pub.publish(pt)
                self.prev_center = p3d.copy()

                # 시각화
                uv_c = self.fusion.rid2img(np.array([p3d], dtype=np.float32))
                if uv_c is not None and len(uv_c) == 1:
                    u, v = int(uv_c[0, 0]), int(uv_c[0, 1])
                    cv2.circle(self.cv_image, (u, v), 6, (0, 255, 0), -1)
                    cv2.putText(self.cv_image, f"conf={conf:.2f}", (u+8, v-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 인라이어 포인트 강조
                if inlier_mask is not None and self.lidar_pts2d is not None and len(self.lidar_pts2d) == len(inlier_mask):
                    for (u, v, _), keep in zip(self.lidar_pts2d, inlier_mask):
                        if keep:
                            cv2.circle(self.cv_image, (int(u), int(v)), 2, (0, 255, 0), -1)

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
