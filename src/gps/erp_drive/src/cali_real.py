#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
import time
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from numpy.linalg import inv
from dynamic_reconfigure.server import Server
from erp_drive.cfg import CaliRealConfig

parameters_cam = {
    "WIDTH": 640, # image width
    "HEIGHT": 480, # image height
    "FX": 651.668905, # focal length x (직접 입력)
    "FY": 652.129398, # focal length y (직접 입력)
    "CX": 327.203220, # principal point x (직접 입력)
    "CY": 242.344293, # principal point y (직접 입력)
    "X": -0.3, # meter
    "Y": 0.,
    "Z":  0.5,
    "YAW": 0.0, #45*math.pi/180.0, # radian
    "PITCH": 0.0,
    "ROLL": 0.0
}
parameters_lidar = {
    "X": 0., # meter
    "Y": 0.,
    "Z": 0.,
    "YAW": 0, # - 7.5*math.pi/180.0, # radian
    "PITCH": 0,
    "ROLL": 0
}
def getRotMat(RPY):
    cosR = math.cos(RPY[0])
    cosP = math.cos(RPY[1])
    cosY = math.cos(RPY[2])
    sinR = math.sin(RPY[0])
    sinP = math.sin(RPY[1])
    sinY = math.sin(RPY[2])
    
    rotRoll = np.array([1,0,0, 0,cosR,-sinR, 0,sinR,cosR]).reshape(3,3)
    rotPitch = np.array([cosP,0,sinP, 0,1,0, -sinP,0,cosP]).reshape(3,3)
    rotYaw = np.array([cosY,-sinY,0, sinY,cosY,0, 0,0,1]).reshape(3,3)
    
    rotMat = rotYaw@rotPitch@rotRoll
    
    return rotMat


def getTransformMat(params_cam, params_lidar):
    #With Respect to Vehicle ISO Coordinate
    lidarPosition = np.array([params_lidar.get(i) for i in (["X","Y","Z"])])
    camPosition = np.array([params_cam.get(i) for i in (["X","Y","Z"])])
    lidarRPY = np.array([params_lidar.get(i) for i in (["ROLL","PITCH","YAW"])])
    camRPY = np.array([params_cam.get(i) for i in (["ROLL","PITCH","YAW"])])
    camRPY = camRPY + np.array([-90*math.pi/180,0,-90*math.pi/180])
    camRot = getRotMat(camRPY)
    camTransl = np.array([camPosition])
    Tr_cam_to_vehicle = np.concatenate((camRot,camTransl.T),axis = 1)
    Tr_cam_to_vehicle = np.insert(Tr_cam_to_vehicle, 3, values=[0,0,0,1],axis = 0)
    lidarRot = getRotMat(lidarRPY)
    lidarTransl = np.array([lidarPosition]) 
    Tr_lidar_to_vehicle = np.concatenate((lidarRot,lidarTransl.T),axis = 1)
    Tr_lidar_to_vehicle = np.insert(Tr_lidar_to_vehicle, 3, values=[0,0,0,1],axis = 0)
    invTr = inv(Tr_cam_to_vehicle)
    Tr_lidar_to_cam = invTr.dot(Tr_lidar_to_vehicle).round(6)
    print(Tr_lidar_to_cam)
    return Tr_lidar_to_cam


def getCameraMat(params_cam):
    # Camera Intrinsic Parameters (직접 입력값 사용)
    fx = params_cam["FX"]
    fy = params_cam["FY"]
    cx = params_cam["CX"]
    cy = params_cam["CY"]
    CameraMat = np.array([fx, 0., cx, 0, fy, cy, 0, 0, 1]).reshape(3,3)
    print("Camera Intrinsic Matrix:")
    print(CameraMat)
    return CameraMat


class LiDARToCameraTransform:
    def __init__(self, params_cam, params_lidar):
        self.scan_sub = rospy.Subscriber("/velodyne_points", PointCloud2, self.scan_callback)
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.img_callback)
        self.image_pub = rospy.Publisher("/lidar_camera_projection", Image, queue_size=1)
        self.bridge = CvBridge()
        self.pc_np = None
        self.img = None
        self.params_cam = params_cam
        self.params_lidar = params_lidar
        self.width = params_cam["WIDTH"]
        self.height = params_cam["HEIGHT"]
        self.TransformMat = getTransformMat(params_cam, params_lidar)
        self.CameraMat = getCameraMat(params_cam)

        # Dynamic reconfigure server
        self.srv = Server(CaliRealConfig, self.dynamic_reconfigure_callback)

    def dynamic_reconfigure_callback(self, config, level):
        # Update camera parameters
        self.params_cam["X"] = config.cam_x
        self.params_cam["Y"] = config.cam_y
        self.params_cam["Z"] = config.cam_z
        self.params_cam["YAW"] = config.cam_yaw * math.pi / 180.0
        self.params_cam["PITCH"] = config.cam_pitch * math.pi / 180.0
        self.params_cam["ROLL"] = config.cam_roll * math.pi / 180.0

        # Recalculate transformation matrix
        self.TransformMat = getTransformMat(self.params_cam, self.params_lidar)
        rospy.loginfo("Updated calibration parameters")
        return config

    def img_callback(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def scan_callback(self, msg):
        point_list = []        
        for point in pc2.read_points(msg, skip_nans=True):      
            point_list.append((point[0], point[1], point[2], 1))
        self.pc_np = np.array(point_list, np.float32)        

    def transformLiDARToCamera(self, pc_lidar):
        cam_temp = self.TransformMat.dot(pc_lidar)
        cam_temp = np.delete(cam_temp, 3, axis=0)
        return cam_temp
    
    def transformCameraToImage(self, pc_camera):
        cam_temp = self.CameraMat.dot(pc_camera)
        cam_temp = np.delete(cam_temp,np.where(cam_temp[2,:]<0),axis=1)
        cam_temp /= cam_temp[2,:]
        cam_temp = np.delete(cam_temp,np.where(cam_temp[0,:]>self.width),axis=1)
        cam_temp = np.delete(cam_temp,np.where(cam_temp[1,:]>self.height),axis=1)
        return cam_temp
    

def draw_pts_img(img, xi, yi):
    point_np = img    
    for ctr in zip(xi, yi):
        point_np = cv2.circle(point_np, ctr, 2, (0,255,0),-1)
    return point_np


if __name__ == '__main__':    
    rospy.init_node('ex_calib', anonymous=True)
    Transformer = LiDARToCameraTransform(parameters_cam, parameters_lidar)
    time.sleep(1)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # Check if image and point cloud data are available
        if Transformer.img is None or Transformer.pc_np is None:
            rate.sleep()
            continue

        xyz_p = Transformer.pc_np[:, 0:3]
        xyz_p = np.insert(xyz_p,3,1,axis=1).T
        xyz_p = np.delete(xyz_p,np.where(xyz_p[0,:]<0),axis=1)  # 뒤쪽 포인트 제거
        xyz_p = np.delete(xyz_p,np.where(xyz_p[0,:]>30),axis=1)  # 30미터 이상 제거 (원래 10)
        xyz_p = np.delete(xyz_p,np.where(xyz_p[2,:]<-1.2),axis=1) #Ground Filter

        #print(xyz_p[0])
        xyz_c = Transformer.transformLiDARToCamera(xyz_p)
        
        #print(np.size(xyz_c[0]))
        xy_i = Transformer.transformCameraToImage(xyz_c)
        
        #print(np.size(xy_i[0]))
        xy_i = xy_i.astype(np.int32)
        projectionImage = draw_pts_img(Transformer.img.copy(), xy_i[0,:], xy_i[1,:])

        # Publish image to ROS topic instead of using cv2.imshow
        try:
            ros_image = Transformer.bridge.cv2_to_imgmsg(projectionImage, encoding="bgr8")
            Transformer.image_pub.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"Failed to publish image: {e}")

        rate.sleep()