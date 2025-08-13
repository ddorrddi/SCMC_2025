import cv2
import numpy as np

# --- 설정 ---
IMAGE_PATH = '/home/ivsp/SCMC_2025_Test/src/camera/bev/scripts/BEV Parking Detection_screenshot_13.08.2025.png'  # 거리를 측정할 BEV 이미지 파일 경로
# --- 설정 끝 ---

# 클릭한 두 점을 저장할 리스트
points = []

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    # 왼쪽 마우스 버튼을 클릭했을 때
    if event == cv2.EVENT_LBUTTONDOWN:
        # 첫 번째 점이면 리스트를 초기화하고 점을 추가
        if len(points) >= 2:
            points.clear()
            # 화면 초기화를 위해 원본 이미지를 다시 복사
            img_display[:] = img_original[:]

        # 클릭한 위치를 points 리스트에 추가
        points.append((x, y))
        
        # 클릭한 위치에 초록색 원 그리기
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        
        # 점이 2개 선택되면 거리 계산 및 표시
        if len(points) == 2:
            pt1 = points[0]
            pt2 = points[1]
            
            # 두 점 사이에 노란색 선 그리기
            cv2.line(img_display, pt1, pt2, (0, 255, 255), 2)
            
            # 유클리드 거리 계산
            pixel_distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            
            # 터미널에 결과 출력
            print("="*30)
            print(f"첫 번째 점 (x, y): {pt1}")
            print(f"두 번째 점 (x, y): {pt2}")
            print(f"두 점 사이의 픽셀 거리: {pixel_distance:.2f} pixels")
            print("="*30)

# 이미지 로드
img_original = cv2.imread(IMAGE_PATH)
if img_original is None:
    print(f"오류: '{IMAGE_PATH}' 파일을 찾을 수 없습니다.")
    exit()

# 화면에 표시될 이미지를 복사 (원본을 보존하기 위함)
img_display = img_original.copy()

# 창 생성 및 콜백 함수 연결
window_name = 'Pixel Distance Measurement'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("이미지 위에서 첫 번째 점을 클릭하세요.")
print("그 다음, 두 번째 점을 클릭하면 두 점 사이의 거리가 계산됩니다.")
print("새로운 측정을 시작하려면 다시 첫 번째 점을 클릭하세요.")
print("'ESC' 키를 누르면 종료됩니다.")

while True:
    cv2.imshow(window_name, img_display)
    # ESC 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()