#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import can
import time
import threading
from race.msg import drive_values

# --- CAN 메시지 ID 정의 ---
ID_AD_CONTROL_FLAG = 0x501
ID_AD_CONTROL_STEERING = 0x502
ID_AD_CONTROL_BRAKE = 0x503
ID_AD_CONTROL_ACCELERATE = 0x504

# --- 글로벌 변수 ---
msg_counter = 0
running = True
bus = None

# 제어 값 저장
current_steering = 0.0
current_velocity = 0.0
current_velocity_sign = 1  # velocity의 부호 (1: 전진, -1: 후진)
current_brake = 0

# --- 스레드 함수 ---

def update_counter():
    """0-15 사이를 순환하는 메시지 카운터를 0.02초마다 업데이트합니다."""
    global msg_counter
    while running:
        msg_counter = (msg_counter + 1) % 16
        time.sleep(0.02)  # 20ms마다 카운터 업데이트

# --- CAN 메시지 생성 함수들 ---

def create_ad_control_flag_msg():
    """자율주행 제어 유효 플래그 메시지 생성"""
    data = [0] * 8
    data[0] = 0x01
    data[0] |= (msg_counter << 4)
    return can.Message(
        arbitration_id=ID_AD_CONTROL_FLAG, data=data, is_extended_id=False
    )

def create_ad_steering_msg(angle_deg=0.0):
    """조향 제어 메시지 생성"""
    data = [0] * 8
    data[0] = 0x01
    data[0] |= (msg_counter << 4)
    can_value = int((angle_deg + 30) / 0.1)
    data[4] = can_value & 0xFF
    data[5] = (can_value >> 8) & 0xFF
    return can.Message(
        arbitration_id=ID_AD_CONTROL_STEERING, data=data, is_extended_id=False
    )

def create_ad_brake_msg(pressure_pct=0):
    """브레이크 제어 메시지 생성"""
    data = [0] * 8
    data[0] = 0x01
    data[0] |= (msg_counter << 4)
    data[1] = pressure_pct
    return can.Message(
        arbitration_id=ID_AD_CONTROL_BRAKE, data=data, is_extended_id=False
    )

def create_ad_accelerate_msg(speed_kph, gear):
    """가속/감속 및 기어 제어 메시지 생성"""
    data = [0] * 8

    # Valid & Counter
    data[0] = 0x01
    data[0] |= (msg_counter << 4)

    # Work Mode: 1 (속도 제어)
    data[2] = 0x01

    # Gear: 입력받은 기어 값 사용
    data[3] = gear

    # Speed Control
    can_value = int(speed_kph / 0.1)
    data[6] = can_value & 0xFF
    data[7] = (can_value >> 8) & 0xFF

    return can.Message(
        arbitration_id=ID_AD_CONTROL_ACCELERATE, data=data, is_extended_id=False
    )

# --- ROS 콜백 함수 ---

def control_value_callback(msg):
    """
    control_value 토픽 콜백 함수
    drive_values 메시지를 받아서 제어값 업데이트
    """
    global current_steering, current_velocity, current_velocity_sign, current_brake

    # steering 값은 이미 deg 단위로 들어옴
    current_steering = msg.steering

    # throttle 처리: 음수면 후진(R), 양수면 전진(D)
    if msg.throttle < 0:
        current_velocity_sign = -1  # 후진
        current_velocity = abs(msg.throttle)
    else:
        current_velocity_sign = 1  # 전진
        current_velocity = msg.throttle

    # brake 처리: 1이면 100%, 0이면 0%
    current_brake = int(msg.brake * 100)
    # print(current_brake)

    gear_str = "R" if current_velocity_sign < 0 else "D"
    rospy.loginfo_throttle(1.0, "Received control_value - Steering: %.2f deg, Velocity: %.2f kph (Gear: %s), Brake: %d%%",
                           current_steering, current_velocity, gear_str, current_brake)

# --- CAN 전송 함수 ---

def send_can_messages():
    """CAN 메시지를 주기적으로 전송하는 함수"""
    global bus, current_steering, current_velocity, current_velocity_sign, current_brake, running

    rate = rospy.Rate(50)
    while not rospy.is_shutdown() and running:
        try:
            # 기어 결정: velocity_sign에 따라 R(3) 또는 D(1)
            if current_velocity_sign < 0:
                gear = 3  # R (후진)
            elif current_velocity > 0:
                gear = 1  # D (전진)
            else:
                gear = 2  # N (중립)

            # 1. 제어 유효 메시지 전송
            bus.send(create_ad_control_flag_msg())

            # 2. 조향 메시지 전송
            bus.send(create_ad_steering_msg(angle_deg=current_steering))

            # 3. 브레이크 메시지 전송
            bus.send(create_ad_brake_msg(pressure_pct=current_brake))

            # 4. 가속/기어 메시지 전송
            accel_msg = create_ad_accelerate_msg(speed_kph=current_velocity, gear=gear)
            bus.send(accel_msg)

        except can.CanError as e:
            rospy.logerr("CAN 에러 발생: %s", e)

        rate.sleep()

# --- 메인 함수 ---

def main():
    global bus, running

    rospy.init_node('control_value_can_controller', anonymous=False)
    rospy.loginfo("control_value_can_controller node started")

    # CAN 인터페이스 설정
    can_interface = rospy.get_param('~can_interface', 'can0')

    try:
        # CAN 버스 초기화
        bus = can.interface.Bus(channel=can_interface, bustype='socketcan')
        rospy.loginfo("CAN bus '%s'에 연결되었습니다.", can_interface)

        # AD 모드 활성화 메시지 전송
        rospy.loginfo("AD 모드 활성화 중...")
        for _ in range(10):
            bus.send(create_ad_control_flag_msg())
            time.sleep(0.02)
        rospy.loginfo("AD 모드 활성화 완료")

        # 카운터 업데이트 스레드 시작
        counter_thread = threading.Thread(target=update_counter, daemon=True)
        counter_thread.start()

        # CAN 전송 스레드 시작
        send_thread = threading.Thread(target=send_can_messages, daemon=True)
        send_thread.start()

        # control_value 토픽 구독
        rospy.Subscriber('control_value', drive_values, control_value_callback)
        rospy.loginfo("control_value 토픽 구독 시작")

        rospy.spin()

    except can.CanError as e:
        rospy.logerr("CAN 에러 발생: %s", e)
        rospy.logerr("'%s' 인터페이스가 활성화되어 있는지 확인해주세요.", can_interface)
        rospy.logerr("예: sudo ip link set can0 up type can bitrate 500000")
    except Exception as e:
        rospy.logerr("예상치 못한 에러: %s", e)
    finally:
        running = False
        if bus:
            rospy.loginfo("안전 정지를 위해 정지 명령(P기어, 속도 0)을 보냅니다.")
            # 기어는 P(0), 속도는 0으로 설정하여 안전하게 정지
            stop_msg = create_ad_accelerate_msg(speed_kph=0.0, gear=0)
            bus.send(stop_msg)
            time.sleep(0.1)
            bus.shutdown()
            rospy.loginfo("CAN bus 연결이 종료되었습니다.")

if __name__ == '__main__':
    main()
