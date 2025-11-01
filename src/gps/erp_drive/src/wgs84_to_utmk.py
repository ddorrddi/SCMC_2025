#!/usr/bin/env python3

# ros 관련 라이브러리
import sys
import rospy

# numpy
import numpy as np

# tranfrom coordinate
from pyproj import Transformer, CRS

# 쿼터니언 <-> 오일러 변환 함수
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from geometry_msgs.msg import Point, Pose, TwistWithCovarianceStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, Int16

import message_filters as mf

from morai_msgs.msg import EgoVehicleStatus
from sensor_msgs.msg import NavSatFix

from erp_drive.srv import InitializePose, InitializePoseResponse

class TransformCoordinate:
    def __init__(self):
        self.is_erp = sys.argv[1] == 'e'

        self.curr_pose_pub = rospy.Publisher('/current_pose', Pose, queue_size=1)
        self.speed_pub = rospy.Publisher("/gps_velocity", Float64, queue_size=1)
        self.utmk_coordinate_pub = rospy.Publisher('/utmk_coordinate', Point, queue_size=1)

        # 방향 전환 트리거 구독
        rospy.Subscriber('/reverse_direction_trigger', Int16, self.reverse_direction_trigger_cb)

        # WGS84
        proj_WGS84 = CRS('EPSG:4326')

        # UTM52N
        proj_UTM52N = CRS('EPSG:32652')

        # UTM-K
        proj_UTMK = CRS('EPSG:5179')

        self.prev_pose = Pose()
        self.prev_pose.position.x = 0
        self.prev_pose.position.y = 0
        
        self.quat = None

        self.curr_pose = Pose()

        if self.is_erp:
            self.transformer = Transformer.from_crs(proj_WGS84, proj_UTMK, always_xy=True)
            rospy.Subscriber('/gps_front/fix', NavSatFix, self.erp_callback)
            rospy.Subscriber('/gps_front/fix_velocity', TwistWithCovarianceStamped, self.velocity_callback)

        else:
            print('hi?')
            self.transformer = Transformer.from_crs(proj_UTM52N, proj_UTMK, always_xy=True)
            rospy.Subscriber('/Ego_topic', EgoVehicleStatus, self.morai_callback)

            # ego_sub = mSubscriber('/Ego_topic', EgoVehicleStatus)
            # imu_sub = mf.Subscriber('/imu', Imu)
            
            # ats = mf.ApproximateTimeSynchronizer([ego_sub, imu_sub], queue_size=10, slop=0.01)
            # ats.registerCallback(self.x_to_utmk)
            
        rospy.Service('InitializePose', InitializePose, self.initialize_curr_pose)
        
    def erp_callback(self, msg: NavSatFix):
        long, lat = msg.longitude, msg.latitude
        self.x_to_utmk(long, lat)

    def velocity_callback(self, msg: TwistWithCovarianceStamped):
        self.xVelocity = msg.twist.twist.linear.x
        self.yVelocity = msg.twist.twist.linear.y
        self.zVelocity = msg.twist.twist.linear.z
        self.speed = np.hypot(self.xVelocity, self.yVelocity) * 3.6

        data = Float64()
        data.data = self.speed
        self.speed_pub.publish(data)

    def morai_callback(self, msg: EgoVehicleStatus):
        x, y = msg.position.x, msg.position.y
        
        x += 302459.942
        y += 4122635.537

        # x += 302595.0
        # y += 4124145.0

        #self.gps_offset_x, self.gps_offset_y = 302595.0, 4124145.0
        
        self.x_to_utmk(x, y)

    def x_to_utmk(self, long, lat): #ego_topic: EgoVehicleStatus, imu: Imu):
        x, y = long, lat

        pub_data = Point(*self.transformer.transform(x, y), 0)
        self.utmk_coordinate_pub.publish(pub_data)
        
        # 이전 좌표와 0.2 이상 차이가 나면 current_pose 갱신 
        prev_x, prev_y = self.prev_pose.position.x, self.prev_pose.position.y
        curr_x, curr_y = pub_data.x, pub_data.y

        self.curr_pose.position.x = curr_x
        self.curr_pose.position.y = curr_y
        self.curr_pose.position.z = 0

        dist = np.hypot(curr_x - prev_x, curr_y - prev_y)
        if dist > 0.2:
            yaw = np.arctan2(curr_y - prev_y, curr_x - prev_x)
            self.quat = quaternion_from_euler(0., 0., yaw)
            self.prev_pose.position.x = self.curr_pose.position.x
            self.prev_pose.position.y = self.curr_pose.position.y
            self.prev_pose.position.z = self.curr_pose.position.z

        self.curr_pose.orientation.x = self.quat[0]
        self.curr_pose.orientation.y = self.quat[1]
        self.curr_pose.orientation.z = self.quat[2]
        self.curr_pose.orientation.w = self.quat[3]
        
        self.curr_pose_pub.publish(self.curr_pose)
        
    def reverse_direction_trigger_cb(self, msg):

        if msg.data != 1:
            return

        curr_x = self.curr_pose.position.x
        curr_y = self.curr_pose.position.y

        if self.quat is None:
            return

        # 현재 yaw 추출
        _, _, curr_yaw = euler_from_quaternion(self.quat)

        # 즉시 yaw를 180도 반전
        new_yaw = curr_yaw + np.pi
        self.quat = quaternion_from_euler(0., 0., new_yaw)

        # prev_pose를 새로운 yaw의 뒤쪽(반대편)에 배치
        # 새 yaw의 반대 = 원래 yaw
        backward_yaw = new_yaw + np.pi  # = curr_yaw + 2π = curr_yaw
        offset_distance = 0.5

        new_prev_x = curr_x + offset_distance * np.cos(backward_yaw)
        new_prev_y = curr_y + offset_distance * np.sin(backward_yaw)

        self.prev_pose.position.x = new_prev_x
        self.prev_pose.position.y = new_prev_y

    def initialize_curr_pose(self):
        self.prev_pose.position.x = 0.0
        self.prev_pose.position.y = 0.0
        self.prev_pose.position.z = 0.0

        self.prev_pose.orientation.x = 0.0
        self.prev_pose.orientation.y = 0.0
        self.prev_pose.orientation.z = 0.0
        self.prev_pose.orientation.w = 0.0

        return InitializePoseResponse(True)

if __name__ == "__main__":
    rospy.init_node("coordinate")
    pub = TransformCoordinate()
    rospy.spin()
