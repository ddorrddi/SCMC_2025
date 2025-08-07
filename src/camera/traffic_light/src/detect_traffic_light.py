#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
from collections import defaultdict

CONFIDENCE = 0.5

# weights_path = '/home/foscar/바탕화면/changyeop_yolo/model/09_09_traffic_light_8n_20epoch.pt'
# weights_path = '/home/foscar/isc  c_2024/src/camera/traffic_light/weight/best (1).pt'
weights_path = '/home/foscar/SCMC_2025_Test/src/camera/traffic_light/weight/1027_40epoch.pt'
model = YOLO(weights_path)
bridge = CvBridge()

# (추가) 처리된 이미지를 발행할 Publisher를 전역 변수로 선언
image_pub = None
traffic_light_pub = None

def image_callback(msg):
    global image_pub # 전역 변수 사용 선언

    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return

    results = model.predict(cv_image, conf=CONFIDENCE)
    results = results[0]
    
    # (수정) preprosses_label 함수가 바운딩 박스가 그려진 이미지를 반환하도록 수정
    final_label, processed_image = preprosses_label(results, cv_image)
    traffic_light_pub.publish(final_label)

    # (수정) cv2.imshow, cv2.waitKey 삭제
    # cv2.imshow('YOLOv8 Traffic Light Detection', cv_image)
    # cv2.waitKey(1)

    # (추가) 처리된 이미지를 ROS 메시지로 변환하여 발행
    if image_pub is not None:
        try:
            img_msg = bridge.cv2_to_imgmsg(processed_image, "bgr8")
            image_pub.publish(img_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")


def preprosses_label(results, cv_image):
    boxes = results.boxes
    names = results.names
    
    # (추가) 탐지된 모든 객체를 초록색으로 먼저 그립니다.
    for i in range(len(boxes)):
        class_index = int(boxes.cls[i])
        class_name = names[class_index]
        confidence = boxes.conf[i]
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        
        # 모든 객체는 초록색으로 표시
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 기존 로직은 그대로 유지하여 최종 신호등을 선택합니다.
    xyxy = boxes.xyxy
    cls = boxes.cls
    confidence = boxes.conf

    height_threshold = 10 
    height_groups = defaultdict(list)

    for i, box in enumerate(xyxy):
        y1 = box[1].item()
        grouped = False
        for key in height_groups.keys():
            if abs(y1 - key) < height_threshold:
                height_groups[key].append(i)
                grouped = True
                break
        if not grouped:
            height_groups[y1] = [i]

    if height_groups:
        highest_group_y = min(height_groups.keys())
        highest_group = height_groups[highest_group_y]
        rightmost_index = max(highest_group, key=lambda i: xyxy[i][2].item())
        
        class_index = int(cls[rightmost_index].item())
        class_name = names[class_index]
        class_confidence = confidence[rightmost_index]
        x1, y1, x2, y2 = map(int, boxes.xyxy[rightmost_index])
        
        # (수정) 최종 선택된 객체만 빨간색으로 덧그려서 강조합니다.
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{class_name}: {class_confidence:.2f}"
        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        print("class name : ", class_name)
        return class_name, cv_image
    else:
        # 탐지된 객체가 있지만, 그룹화 로직에서 선택되지 않은 경우, 초록색 박스가 그려진 이미지를 반환
        return "", cv_image


def yolo_traffic_light_node():
    rospy.init_node('yolo_traffic_light_node', anonymous=True)

    # (수정) Publisher들을 전역 변수로 사용하도록 선언
    global traffic_light_pub, image_pub

    rospy.Subscriber('/usb_cam/image_raw', Image, image_callback, queue_size=1) # queue_size=1 추가

    traffic_light_pub = rospy.Publisher('/traffic_light_status', String, queue_size=10)
    
    # (추가) 처리된 이미지를 발행할 Publisher 초기화
    image_pub = rospy.Publisher('/yolo_traffic_light/image_processed', Image, queue_size=1)

    rospy.spin()

if __name__ == '__main__':
    try:
        yolo_traffic_light_node()
    except rospy.ROSInterruptException:
        pass