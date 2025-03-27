#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
from ultralytics import YOLO

CONFIDENCE = 0.5

weights_path = '/home/foscar/iscc_2024/src/camera/traffic_light/weight/yolo8m_traffic_light.pt'
model = YOLO(weights_path)
bridge = CvBridge()

def image_callback(msg):
    try:
        cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return

    results = model.predict(cv_image, conf=CONFIDENCE)
    results = results[0]
    boxes = results.boxes
    names = results.names
    for i in range(len(boxes)):
        class_index = int(boxes.cls[i])
        class_name = names[class_index]
        confidence = boxes.conf[i]
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])

        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        traffic_light_pub.publish(label)
        
    cv2.imshow('YOLOv8 Traffic Light Detection', cv_image)
    cv2.waitKey(1)

def yolo_traffic_light_node():
    rospy.init_node('yolo_traffic_light_node', anonymous=True)

    rospy.Subscriber('/image_jpeg/compressed', CompressedImage, image_callback)

    global traffic_light_pub
    traffic_light_pub = rospy.Publisher('/traffic_light_status', String, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    try:
        yolo_traffic_light_node()
    except rospy.ROSInterruptException:
        pass
