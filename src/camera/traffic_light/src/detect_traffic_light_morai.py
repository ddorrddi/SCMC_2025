#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
from ultralytics import YOLO

CONFIDENCE = 0.4

weights_path = '/home/foscar/SCMC_2025/src/camera/traffic_light/weight/1027_40epoch.pt'
model = YOLO(weights_path)
bridge = CvBridge()

# **(수정)** 처리된 이미지를 발행할 Publisher를 전역 변수로 선언
image_pub = None

def image_callback(msg):
    try:
        cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return

    results = model.predict(cv_image, conf=CONFIDENCE, verbose=False)
    results = results[0]
    boxes = results.boxes
    names = results.names

    # 감지된 객체가 있을 때만 이미지에 그리고 발행하도록 수정
    if len(boxes) > 0:
        # Find the largest bounding box by area
        max_area = 0
        largest_idx = 0
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_idx = i

        # Draw all detections but only publish the largest one
        for i in range(len(boxes)):
            class_index = int(boxes.cls[i])
            class_name = names[class_index]
            confidence = boxes.conf[i]
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])

            # Use different color for the largest box
            color = (0, 0, 255) if i == largest_idx else (0, 255, 0)  # Red for largest, green for others
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}"
            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Only publish the largest detection
            if i == largest_idx:
                traffic_light_pub.publish(label)
                rospy.loginfo(f"[TRAFFIC] Publishing largest: {label} (area: {max_area})")

    # **(수정)** cv2.imshow, cv2.waitKey 삭제
    # cv2.imshow('YOLOv8 Traffic Light Detection', cv_image)
    # cv2.waitKey(1)

    # **(추가)** 처리된 이미지를 ROS 메시지로 변환하여 발행
    try:
        img_msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        image_pub.publish(img_msg)
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")


def yolo_traffic_light_node():
    rospy.init_node('yolo_traffic_light_node', anonymous=True)

    # **(수정)** image_pub를 전역 변수로 사용
    global traffic_light_pub, image_pub

    rospy.Subscriber('/image_jpeg/compressed', CompressedImage, image_callback, queue_size=1)

    traffic_light_pub = rospy.Publisher('/traffic_light_status', String, queue_size=10)
    # **(추가)** 처리된 이미지를 발행할 Publisher 초기화
    image_pub = rospy.Publisher('/yolo_traffic_light/image_raw', Image, queue_size=1)


    rospy.spin()

if __name__ == '__main__':
    try:
        yolo_traffic_light_node()
    except rospy.ROSInterruptException:
        pass