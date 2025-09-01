import numpy as np


class Fusion:
    def __init__(self, Fx, Fy, Cx, Cy, R, T, img_width, img_height): # 파라미터 K, R, t 초기화
        # 내부 파라미터 K
        self.Fx = Fx
        self.Fy = Fy
        self.Cx = Cx
        self.Cy = Cy
        
        # 외부 파라미터 R (라이다-카메라)
        self.R = np.array(R).reshape(3,3) # 3x3

        # 외부 파라미터 t (라이다-카메라)
        self.T = np.array(T).reshape(3,1) # 3x1

        # 카메라 내부 파라미터
        self.K = np.array([
            [Fx, 0, Cx],
            [0, Fy, Cy],
            [0,  0,  1]
        ])

        # Extrinsic
        self.Rt = np.hstack((self.R, self.T))

        # 카메라 해상도
        self.img_width = img_width
        self.img_height = img_height
 
    @staticmethod
    def rpy_to_matrix(roll, pitch, yaw):
        """Roll, Pitch, Yaw(rad) → 회전행렬 (Z-Y-X 순서)"""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    @staticmethod
    def make_extrinsics(x, y, z, roll, pitch, yaw, convention="velodyne_to_cam"):
        """
        라이다 → 카메라 외부 파라미터(R, T) 생성
        x, y, z: [m]
        roll, pitch, yaw: [rad]
        convention: 좌표계 축 변환 (default: Velodyne → Camera(OpenCV))
        """
        R_rpy = Fusion.rpy_to_matrix(roll, pitch, yaw)

        if convention == "velodyne_to_cam":
            # Velodyne(+X forward, +Y left, +Z up)
            # Camera  (+X right, +Y down, +Z forward)
            R_axes = np.array([
                [0, -1, 0],   # X_cam = -Y_lidar
                [0,  0, -1],  # Y_cam = -Z_lidar
                [1,  0,  0]   # Z_cam =  X_lidar
            ])
        else:
            R_axes = np.eye(3)

        R = R_axes @ R_rpy
        T = np.array([[x], [y], [z]])
        return R, T


    # @staticmethod
    # def make_extrinsics(
    #     x, y, z, roll, pitch, yaw,
    #     convention="velodyne_to_cam",
    #     angles_in="lidar",      # "lidar" or "camera"
    #     trans_in="lidar",       # "lidar" or "camera"
    #     rpy_order="zyx"         # yaw-pitch-roll(=ZYX) 권장
    # ):
    #     """
    #     Returns R, T for p_cam = R * p_lidar + T

    #     x,y,z:
    #     - trans_in=="lidar":  LiDAR축 기준의 'LiDAR→Camera' 벡터 d_l (O_c - O_l)
    #     - trans_in=="camera": 카메라축 기준의 LiDAR 원점 위치 T (즉, 그대로 T)

    #     roll,pitch,yaw:
    #     - angles_in=="lidar": LiDAR축 기준 추가 회전
    #     - angles_in=="camera": 카메라축 기준 추가 회전

    #     convention:
    #     "velodyne_to_cam" => 고정 축치환( LiDAR( +X fwd, +Y left, +Z up ) → Camera( +X right, +Y down, +Z fwd ) )
    #     """
    #     # 1) 추가 오차 회전 R_rpy
    #     R_rpy = Fusion.rpy_to_matrix(roll, pitch, yaw, order=rpy_order)

    #     # 2) 축 치환 행렬 A (LiDAR basis -> Camera basis)
    #     if convention == "velodyne_to_cam":
    #         A = np.array([
    #             [0, -1,  0],  # X_cam = -Y_lidar
    #             [0,  0, -1],  # Y_cam = -Z_lidar
    #             [1,  0,  0],  # Z_cam =  X_lidar
    #         ], dtype=float)
    #     else:
    #         A = np.eye(3)

    #     # 3) 전체 회전 R_total 구성
    #     if angles_in == "lidar":
    #         # LiDAR축에서 오차회전을 준 다음 카메라축으로 치환
    #         R = A @ R_rpy
    #     elif angles_in == "camera":
    #         # 먼저 카메라축으로 치환한 뒤 카메라축에서 오차회전
    #         R = R_rpy @ A
    #     else:
    #         raise ValueError("angles_in must be 'lidar' or 'camera'")

    #     # 4) 평행이동 T 계산
    #     tvec = np.array([x, y, z], dtype=float).reshape(3)

    #     if trans_in == "lidar":
    #         # 입력은 d_l = (O_c - O_l) expressed in LiDAR frame
    #         # 카메라 좌표로 변환: d_c = R * d_l
    #         d_c = R @ tvec
    #         # T = O_l (in camera) = -d_c
    #         T = (-d_c).reshape(3, 1)
    #     elif trans_in == "camera":
    #         # 이미 카메라좌표계에서의 LiDAR 원점 위치라면 그대로 T
    #         T = tvec.reshape(3, 1)
    #     else:
    #         raise ValueError("trans_in must be 'lidar' or 'camera'")

    #     return R, T


    def update_extrinsics(self, R, T):
        self.R = R
        self.T = T
        self.Rt = np.hstack((self.R, self.T))

    def getInvMat(self, mat:np.ndarray)->np.ndarray: # 역행렬 리턴
        return np.linalg.inv(mat)


    def rid2img(self, points_ridar:np.ndarray, return_mask:bool=False)->np.ndarray:
        """라이다 좌표계 점들(N,3)을 이미지 좌표(N',2)로 투영.
        return_mask=True이면 입력 배열에 대한 유효 인덱스(np.int64)를 함께 반환.
        """
        if points_ridar is None or len(points_ridar) == 0:
            raise ValueError("empty points_ridar")
        if points_ridar.shape[1] != 3:
            raise ValueError("points_ridar is not (N,3)")
        
        k = self.K
        rt = self.Rt

        N = points_ridar.shape[0]
        points_homo = np.hstack([points_ridar, np.ones((N, 1))])   # [N,4]
        points_cam = (rt @ points_homo.T).T                        # [N,3]

        # 카메라 뒤쪽(Z<=0) 제거
        cam_mask = points_cam[:, 2] > 0                            # [N]
        if not np.any(cam_mask):
            if return_mask:
                return np.empty((0, 2), dtype=np.float32), np.array([], dtype=np.int64)
            return np.empty((0, 2), dtype=np.float32)
        points_cam = points_cam[cam_mask]                          # [M,3]

        # 투영
        points_img = (k @ points_cam.T).T                          # [M,3]

        # 정규화
        points_2d = np.empty((points_cam.shape[0], 2), dtype=np.float32)
        points_2d[:, 0] = points_img[:, 0] / points_img[:, 2]
        points_2d[:, 1] = points_img[:, 1] / points_img[:, 2]

        # 이미지 경계 clip
        points_2d[:, 0] = np.clip(points_2d[:, 0], 0, self.img_width - 1)
        points_2d[:, 1] = np.clip(points_2d[:, 1], 0, self.img_height - 1)

        if return_mask:
            idx = np.nonzero(cam_mask)[0]  # ← 원본(N)에서 살아남은 M개의 인덱스
            return points_2d, idx

        return points_2d



    def points2bbox(self, points_2d:np.ndarray)->np.ndarray: # 이미지상의 점들을 입력으로 받아 모든 점들을 가두는 BoundingBox행렬을 만든다.
        if points_2d.size == 0:
            raise ValueError("empty points_2d")
    
        u_min = np.min(points_2d[:, 0])
        u_max = np.max(points_2d[:, 0])
        v_min = np.min(points_2d[:, 1])
        v_max = np.max(points_2d[:, 1])

        bbox = np.array([
            [u_min, v_min], # 좌측상단 점
            [u_max, v_min], # 우측상단 점
            [u_max, v_max], # 우측하단 점
            [u_min, v_max]  # 좌측하단 점
        ])
        return bbox
    

    def bbox_iou(self, bbox1:np.ndarray, bbox2:np.ndarray)->float:
        """
        bbox: (n,2) np배열, 꼭짓점 좌표들 (u,v)
        두 bbox의 IoU 계산 (axis-aligned bounding box로 간주)
        """
        # 각 bbox에서 u,v 최소/최대 구하기
        u_min1, v_min1 = np.min(bbox1, axis=0)
        u_max1, v_max1 = np.max(bbox1, axis=0)
        
        u_min2, v_min2 = np.min(bbox2, axis=0)
        u_max2, v_max2 = np.max(bbox2, axis=0)

        # 각 bbox 넓이 계산
        area1 = (u_max1 - u_min1) * (v_max1 - v_min1)
        area2 = (u_max2 - u_min2) * (v_max2 - v_min2)
        
        # 교집합 영역 계산
        inter_u_min = max(u_min1, u_min2)
        inter_v_min = max(v_min1, v_min2)
        inter_u_max = min(u_max1, u_max2)
        inter_v_max = min(v_max1, v_max2)
        
        inter_width = max(0, inter_u_max - inter_u_min)
        inter_height = max(0, inter_v_max - inter_v_min)
        inter_area = inter_width * inter_height
        
        # 합집합 영역 계산
        union_area = area1 + area2 - inter_area

        # 둘중 하나라도 넓이가 0아래면 0.0반환 / 합집합의 넓이가 0아래면 0.0반환 (0으로 나누기 방지)
        if area1 <= 0 or area2 <= 0 or union_area <= 0:
            return 0.0
        
        # IoU 계산
        iou = inter_area / union_area
        return iou
