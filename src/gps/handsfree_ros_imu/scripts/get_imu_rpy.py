#!/usr/bin/env python
#coding=UTF-8

import rospy, math
from tf.transformations import *
from sensor_msgs.msg import Imu

def callback(data):
    r, p, y = euler_from_quaternion((data.orientation.x,data.orientation.y,data.orientation.z,data.orientation.w))
    rospy.loginfo("Pitch = %f,Roll = %f, Yaw = %f",r*180/math.pi,p*180/math.pi,y*180/math.pi)

def get_imu():
    rospy.init_node('get_imu', anonymous=True)
    rospy.Subscriber("/handsfree/imu", Imu, callback)
    rospy.spin()

if __name__ == '__main__':
    get_imu()
