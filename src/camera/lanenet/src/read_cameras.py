import cv2

# 웹캠 캡처 객체 생성 (기본 웹캠은 인덱스 0)
cap = cv2.VideoCapture(0)

# 비디오 코덱 설정 (XVID 코덱을 사용)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 출력 파일 설정 (파일명, 코덱, FPS, 프레임 크기)
out = cv2.VideoWriter('output_delivery.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임을 파일에 기록
    out.write(frame)

    # 프레임을 윈도우에 표시
    cv2.imshow('frame', frame)

    # 'q' 키를 누르면 녹화 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
