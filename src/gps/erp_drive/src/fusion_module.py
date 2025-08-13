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

        # K, Rt np행렬
        self.K = self.getKMat()
        self.Rt = self.getRtMat()

        # 카메라 해상도
        self.img_width = img_width
        self.img_height = img_height


    def getKMat(self)->np.ndarray: # K행렬 리턴 (3x3)
        K = np.array([
            [self.Fx,       0, self.Cx],
            [      0, self.Fy, self.Cy],
            [      0,       0,       1]
        ])
        return K
    

    def getRtMat(self)->np.ndarray: # R|t 행렬 리턴 (3x4)
        Rt = np.hstack((self.R, self.T)).reshape(3,4)
        return Rt
    

    def getInvMat(self, mat:np.ndarray)->np.ndarray: # 역행렬 리턴
        return np.linalg.inv(mat)


    def rid2img(self, points_ridar:np.ndarray)->np.ndarray: # 라이다좌표계의 점[[X,Y,Z], ...]들을 이미지상의[[u,v], ...]로 바꿔주는 함수
        if points_ridar is None or len(points_ridar) == 0: # 빈 points_ridar 방지
            raise ValueError("empty points_ridar")

        if points_ridar.shape[1] != 3: # 형태가 N,3형태가 아닌경우 방지
            raise ValueError("points_ridar is not (N,3)")
        
        points_2d = [] # 결과값
        k = self.K # 내부 파라미터
        rt = self.Rt # 외부 파라미터

        # for문 기반은 속도가 느리기때문에 벡터연산 기반으로 바꿈
        # for point_ridar in points_ridar: # [[X,Y,Z], ...]
        #     point_4d = np.append(point_ridar, 1).reshape(4,1) # 4x1
        #     point_cam = rt @ point_4d # 3x1
        #     point_img = k @ point_cam # 3x1
        #     # 정규화
        #     u = point_img[0][0] / point_img[2][0] # u/w
        #     v = point_img[1][0] / point_img[2][0] # v/w
        #     points_2d.append([u,v])
        # points_2d = np.array(points_2d) # [[u,v], ...] 리스트를 np행렬로 변환

        N = points_ridar.shape[0] # 라이다로 인지한 포인트들의 개수
        points_homo = np.hstack([points_ridar, np.ones((N, 1))])  # [N,4] ← 4차 동차좌표로 확장
        points_cam = (rt @ points_homo.T).T                       # [N,3] ← 카메라 좌표계로 변환

        # 카메라 뒤쪽 (Z <= 0) 제거
        mask = points_cam[:, 2] > 0                               # [N] ← Z축이 양수인 점만
        points_cam = points_cam[mask]                             # [M,3]

        """"""
        print("[points_cam]")
        print(points_cam)
        """"""

        if points_cam.shape[0] == 0:
            return np.empty((0, 2))

        points_img = (k @ points_cam.T).T                         # [M,3] ← 이미지 평면으로 변환 (투영)]
        """"""
        print("[points_img]")
        print(points_img)
        """"""

        u = points_img[:, 0] / points_img[:, 2]                   # [M] ← u 좌표 정규화
        v = points_img[:, 1] / points_img[:, 2]                   # [M] ← v 좌표 정규화
        points_2d = np.stack([u, v], axis=1)                      # [M,2] ← 2D 이미지 좌표 조합

        """"""
        print("[points_2d]")
        print(points_2d)
        """"""

        # 이미지 해상도에 맞게 clip
        points_2d[:, 0] = np.clip(points_2d[:, 0], 0, self.img_width-1)
        points_2d[:, 1] = np.clip(points_2d[:, 1], 0, self.img_height-1)

        points_2d.astype(np.float32)/1000.0

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
