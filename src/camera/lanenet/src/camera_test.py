import cv2

def main():
    # 첫 번째 카메라 장치 열기
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        w, h, c = frame.shape
        print(w)
        print(h)
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 프레임을 윈도우에 표시
        cv2.imshow('USB Camera', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 모든 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
