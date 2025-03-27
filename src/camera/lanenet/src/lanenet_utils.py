import numpy as np
import cv2, math
import rospy, rospkg, time

from cv_bridge import CvBridge

from math import *
import signal
import sys
import os
import random



previous_angle = 0
show_img = np.empty(shape=[0])

# 추후에 color filtering을 통해 흰색의 차선만을 추출하기 위한 픽셀 범위값을 미리 선언하였습니다
global lower_white
lower_white = np.array([200,60,150])
global upper_white
upper_white = np.array([255,255,255])
global yellow_lower
lower_yellow = np.array([200,60,150])
global yellow_upper
upper_yellow = np.array([255,255,255])


def init_show_img(img) :
    global show_img
    show_img = img


def signal_handler(sig, frame):
    time.sleep(3)
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


image = np.empty(shape=[0]) 
bridge = CvBridge() 
motor = 0


CAM_FPS = 30    
WIDTH, HEIGHT = 640, 480    


def img_callback(data):
    
    
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

global motor_info


def color_filter(img):
    mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(img,lower_white,upper_white)
    masks = cv2.bitwise_or(mask_yellow, mask_white)
    masked = cv2.bitwise_and(img,img,mask = masks)

    return masked


# 카메라의 원근왜곡을 제거하여 차선을 평행하게 만들어주기 위한 Bird-eye-view 변환 함수입니다.
# bird-eye-view 변환에는 넓은시야각을 얻기 위해 역투영변환법을 적용하였습니다
def bird_eye_view(img,width,height):
    src = np.float32([[0,0],
                  [width,0],
                  [0,160],
                  [width,160]])

    dst = np.float32([[0,0],
                  [width,0],
                  [185,height],
                  [460,height]])   #src, dst는 모두 순서대로 왼쪽위, 오른쪽위, 왼쪽아래, 오른쪽 아래 점입니다.230, 415
                                    #원본의 점인 src를 상단보다 하단이 좁은 사다리꼴 형태의 dst점으로 변환시킴으로서 bird-eye-view가 완성됩니다.
         
    M = cv2.getPerspectiveTransform(src,dst)
    M_inv = cv2.getPerspectiveTransform(dst,src)
    img_warped = cv2.warpPerspective(img,M,(width,height))  # cv2.getPerspectiveTransform 함수를 이용하여 변환행렬을 계산하고 warpPerspective 함수를 이용하여 시점변환하였습니다.
    return img_warped


#차선이 포함될 것으로 예상되는 부분 이외의 픽셀의 픽셀값은 모두 0으로 바꿔주었습니다. 차량이 속해있는 차선외의 차선에 간섭을 줄이기 위함입니다.
def region_of_interest(img):
    #height = 480
    # img=cv2.resize(img, dsize=(640,480))

    height = 480
    width = 640
    mask = np.zeros((height,width),dtype="uint8")
    pts = np.array([[100,0],[500,0],[500,480],[100,480]])  # 차례대로 왼쪽위, 오른쪽위, 오른쪽아래,왼쪽아래 점이며 저희는 하나의 차선만 인식되어도 안정적으로 주행할 수 있도록 구현하였기에 roi를 보수적으로 설정하여 노이즈를 가능한 줄여주었습니다.
    mask= cv2.fillPoly(mask,[pts],(255,255,255),cv2.LINE_AA)


    img_masked = cv2.bitwise_and(img,mask)
    return img_masked


#차선의 곡률반경을 구하는 함수입니다.
#자세한 계산방법은 과제3번 SW설계서 1번미션에 기술하였습니다.
#차선의 맨 윗점과 맨 아랫점을 추출하고, 두 점 사이의 거리와 두점을 잇는 직선의 방정식을 이용하여 계산하였습니다.
def Radius(lx, ly):
    a=0
    b=0
    c=0
    R=0
    h=0
    w=0  #변수들을 초기화하는 과정입니다.
    if (lx[-1] - lx[1] != 0): 
        a = (ly[-1] - ly[3]) / (lx[-1] - lx[3])
        b = -1
        c = ly[3] - lx[3] * (ly[-1] - ly[3]) / (lx[-1] - lx[3])
        h = abs(a * np.mean(lx) + b * np.mean(ly) + c) / math.sqrt(pow(a, 2) + pow(b, 2))
        w = math.sqrt(pow((ly[-1] - ly[3]), 2) + pow((lx[-1] - lx[3]), 2))
             #rx,ry 는 각 window 조사창 내에 속해있는 흰색 픽셀들의 픽셀좌표의 평균을 담아놓은 리스트입니다.
	     #rx[-1]은 제일 위에있는 window를, rx[3]은 아래에서 4번째에 있는 window를 의미합니다.
	     #rx[0]대신 rx[3]을 이용한 이유는 시뮬레이터상의 카메라 높이가 낮아 차량에 가까운 차선이 인식이 불안정하였기 때문입니다.

    if h != 0:
        R = h / 2 + pow(w, 2) / h * 8
	
    return R*3/800 