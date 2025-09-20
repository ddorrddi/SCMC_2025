import numpy as np
from typing import Optional


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

    @staticmethod
    def robust_mad_sigma(x: np.ndarray, eps: float = 1e-6):
        """로버스트 분산 추정: sigma ≈ 1.4826 * MAD"""
        if x.size == 0:
            return 0.0, 1.0
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        sigma = 1.4826 * mad + eps
        return med, sigma

    def compute_target_from_bbox(
        self,
        cloud_uv: np.ndarray,      # (N,2) float32
        cloud_xyz: np.ndarray,     # (N,3) float32 (LiDAR frame)
        xyxy: tuple,               # (x1,y1,x2,y2)
        r_cut: float = 0.8,        # bbox 중심성 컷(정규 반경)
        sigma2d: float = 0.5,      # 중심 가중치 분산
        k_mad: float = 2.5,        # 깊이(MAD) 이상치 컷 강도
        neigh_radius_base: float = 0.25,    # 근접도(반경-이웃) 기준
        neigh_radius_scale: float = 0.02,   # 거리 비례 반경 스케일
        neigh_min_samples: int = 3,         # 최소 이웃 수
        prev_center: Optional[np.ndarray] = None,  # (3,) 이전 프레임 중심 (옵션)
        prev_gate: Optional[float] = 1.5,          # 시간 게이팅 반경 (옵션)
        tau_z2: float = 0.36,    # 신뢰도 계산용 깊이 분산 스케일 (0.6m)^2
        tau_2d: float = 0.25,    # 신뢰도 계산용 중심성 스케일
        max_points: int = 512,   # 부하 제한 (너무 많으면 다운샘플)
        max_range: float = 10.0  # [m], 필요에 따라 파라미터화 가능
    ):
        """
        YOLO bbox(xyxy) 내부의 투영 포인트(cloud_uv)와 그에 대응하는 LiDAR 3D(cloud_xyz)에서
        노이즈를 정제하고 가중 평균 3D 좌표를 반환.

        return:
            p_mean: (3,) float or None
            conf:  float (0~1)
            inlier_mask: (N,) bool (cloud_uv/xyz 길이와 동일, 최종 인라이어 표시) 또는 None
        """
        # 유효성
        if cloud_uv is None or cloud_xyz is None:
            return None, 0.0, None
        if len(cloud_uv) == 0 or len(cloud_xyz) == 0 or len(cloud_uv) != len(cloud_xyz):
            return None, 0.0, None

        x1, y1, x2, y2 = xyxy
        u = cloud_uv[:, 0]
        v = cloud_uv[:, 1]

        # 1) bbox 내부 선택
        inb = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        if not np.any(inb):
            return None, 0.0, np.zeros(len(cloud_uv), dtype=bool)

        P  = cloud_xyz[inb]         # (M,3)
        UV = cloud_uv[inb]          # (M,2)

        # 너무 많으면 다운샘플
        if len(P) > max_points:
            sel = np.random.choice(len(P), max_points, replace=False)
            P, UV = P[sel], UV[sel]
            # inlier mask는 마지막에 좌표 매칭으로 복원

        # 2) 중심성 컷/가중치 (bbox 기준 정규 반경 r)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        r = np.sqrt(((UV[:, 0] - cx) / w) ** 2 + ((UV[:, 1] - cy) / h) ** 2)

        keep = (r <= r_cut)
        if not np.any(keep):
            return None, 0.0, np.zeros(len(cloud_uv), dtype=bool)
        P, UV, r = P[keep], UV[keep], r[keep]
        w2d = np.exp(-(r ** 2) / (sigma2d ** 2))

        # 3) 거리(범위) 기반 로버스트 게이팅: Median + MAD
        R = np.linalg.norm(P, axis=1)
        med, sigma = self.robust_mad_sigma(R)
        keep = np.abs(R - med) <= (k_mad * sigma)
        if not np.any(keep):
            return None, 0.0, np.zeros(len(cloud_uv), dtype=bool)
        P, R, w2d, r = P[keep], R[keep], w2d[keep], r[keep]

        # === 3.5) 거리 임계값 컷오프 ===
        keep = R <= max_range
        if not np.any(keep):
            return None, 0.0, np.zeros(len(cloud_uv), dtype=bool)
        P, R, w2d, r = P[keep], R[keep], w2d[keep], r[keep]

        # 4) 3D 근접도 코어 선택(반경-이웃)
        rad = max(neigh_radius_base, neigh_radius_scale * max(med, 1.0))
        D = np.linalg.norm(P[None, :, :] - P[:, None, :], axis=2)  # (K,K)
        neigh = (D <= rad).sum(axis=1)
        core = neigh >= neigh_min_samples
        if not np.any(core):
            # 코어 없으면 이웃 많은 상위 5개 사용
            topk = min(5, len(P))
            idx = np.argsort(-neigh)[:topk]
            core = np.zeros(len(P), dtype=bool)
            core[idx] = True

        P, R, w2d, r = P[core], R[core], w2d[core], r[core]
        if len(P) == 0:
            return None, 0.0, np.zeros(len(cloud_uv), dtype=bool)

        # 5) 시간 일관성(옵션)
        if prev_center is not None and prev_gate is not None:
            d_prev = np.linalg.norm(P - prev_center[None, :], axis=1)
            tmask = d_prev <= prev_gate
            if np.any(tmask):
                P, R, w2d, r = P[tmask], R[tmask], w2d[tmask], r[tmask]
                if len(P) == 0:
                    return None, 0.0, np.zeros(len(cloud_uv), dtype=bool)

        # 6) 깊이 가중치 + 최종 가중 평균
        sigma_eff = max(sigma, 0.05)  # 너무 작아지지 않도록 하한
        wz = np.exp(-((R - med) ** 2) / (2.0 * sigma_eff ** 2))
        w_all = w2d * wz
        ws = float(np.sum(w_all))
        if ws <= 1e-6:
            return None, 0.0, np.zeros(len(cloud_uv), dtype=bool)
        p_mean = (w_all[:, None] * P).sum(axis=0) / ws  # (3,)

        # 7) 신뢰도 추정
        var_z = float(np.average((R - np.average(R, weights=w_all)) ** 2, weights=w_all))
        r2bar = float(np.average(r ** 2, weights=w2d))
        frac  = float(len(P)) / float(np.count_nonzero(inb))
        conf  = float(np.clip(frac, 0, 1) * np.exp(-var_z / tau_z2) * np.exp(-r2bar / tau_2d))

        # 8) 최종 인라이어 마스크 (원본 길이와 동일하게 복원)
        final_mask = np.zeros(len(cloud_uv), dtype=bool)
        # inb → center → MAD → core(→time) 경로를 좌표 매칭으로 표시
        idx_inb = np.where(inb)[0]

        # 다운샘플/필터를 거쳐 남은 P를 원본 인덱스로 매칭
        # (포인트 수가 비교적 작으므로 근사 매칭으로 충분)
        base = cloud_xyz[idx_inb]
        for p in P:
            j = int(np.argmin(np.linalg.norm(base - p[None, :], axis=1)))
            final_mask[idx_inb[j]] = True

        return p_mean.astype(np.float32), conf, final_mask

    def update_extrinsics(self, R, T):
        self.R = R
        self.T = T
        self.Rt = np.hstack((self.R, self.T))

    def getInvMat(self, mat:np.ndarray)->np.ndarray: # 역행렬 리턴
        return np.linalg.inv(mat)


    def rid2img(self, points_ridar:np.ndarray, return_mask:bool=False, clip: bool=False)->np.ndarray:
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
        if clip:
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
