#!/usr/bin/env python

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

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return

    results = model.predict(cv_image, conf=CONFIDENCE)
    results = results[0]
    boxes = results.boxes
    names = results.names
    final_label, cv_image = preprosses_label(results, cv_image)
    traffic_light_pub.publish(final_label)

    # for i in range(len(boxes)):
    #     class_index = int(boxes.cls[i])
    #     class_name = names[class_index]
    #     confidence = boxes.conf[i]
    #     x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        
    #     cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     label = f"{class_name}: {confidence:.2f}"
    #     cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imshow('YOLOv8 Traffic Light Detection', cv_image)
    cv2.waitKey(1)


def preprosses_label(results, cv_image):
    boxes = results.boxes
    names = results.names
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

    # height_groups 예시
    # {y 좌표 : 바운딩 박스 인덱스 값}
    # height_groups = {
    # 215.0987: [0, 1, 4], 
    # 314.4227: [2, 3]      
    # }  

    if height_groups:
        highest_group_y = min(height_groups.keys())
        highest_group = height_groups[highest_group_y]

        rightmost_index = max(highest_group, key=lambda i: xyxy[i][2].item())
        
        class_index = int(cls[rightmost_index].item())
        class_name = names[class_index]
        class_confidence = confidence[rightmost_index]

        x1, y1, x2, y2 = map(int, boxes.xyxy[rightmost_index])
        
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{class_name}: {class_confidence:.2f}"
        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print("class name : ", class_name)
        return class_name, cv_image
    else:
        return "", cv_image


def yolo_traffic_light_node():
    rospy.init_node('yolo_traffic_light_node', anonymous=True)

    rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

    global traffic_light_pub
    traffic_light_pub = rospy.Publisher('/traffic_light_status', String, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    try:
        yolo_traffic_light_node()
    except rospy.ROSInterruptException:
        pass
