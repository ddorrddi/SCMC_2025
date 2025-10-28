import can
import time
import threading

# --- CAN 메시지 ID 정의 ---
ID_AD_CONTROL_FLAG = 0x501
ID_AD_CONTROL_STEERING = 0x502
ID_AD_CONTROL_BRAKE = 0x503
ID_AD_CONTROL_ACCELERATE = 0x504

# --- 글로벌 변수 ---
msg_counter = 0
running = True

# --- 스레드 함수 ---

def update_counter():
    """0-15 사이를 순환하는 메시지 카운터를 0.02초마다 업데이트합니다."""
    global msg_counter
    while running:
        msg_counter = (msg_counter + 1) % 16
        time.sleep(0.02) # 20ms마다 카운터 업데이트

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

# --- 메인 실행 함수 ---

def main():
    """메인 실행 함수"""
    can_interface = 'can0'
    bus = None

    # ======================================================
    # ▼▼▼▼▼▼▼▼▼▼▼▼ 여기서 목표 값 설정 ▼▼▼▼▼▼▼▼▼▼▼▼
    # ======================================================
    target_speed_kph = 1.0  # 목표 속도 (km/h)
    
    # 목표 기어 설정 (0:P, 1:D, 2:N, 3:R)
    target_gear = 1         # 1로 설정하여 'D' (주행) 모드로 시작
    # ======================================================
    
    gear_map = {0: 'P', 1: 'D', 2: 'N', 3: 'R'}

    try:
        bus = can.interface.Bus(channel=can_interface, bustype='socketcan')
        print(f"CAN bus '{can_interface}'에 연결되었습니다.")

        # 카운터 업데이트 스레드 시작
        counter_thread = threading.Thread(target=update_counter, daemon=True)
        counter_thread.start()

        print(f"차량 제어를 시작합니다.")
        print(f"목표 속도: {target_speed_kph} km/h, 목표 기어: {gear_map.get(target_gear, 'Unknown')}")
        
        while True:
            # 1. 제어 유효 메시지 전송
            bus.send(create_ad_control_flag_msg())
            
            # 2. 조향 메시지 전송 (직진)
            bus.send(create_ad_steering_msg(angle_deg=-10.0))

            # 3. 브레이크 메시지 전송 (브레이크 해제)
            bus.send(create_ad_brake_msg(pressure_pct=0))

            # 4. 가속/기어 메시지 전송
            accel_msg = create_ad_accelerate_msg(speed_kph=target_speed_kph, gear=target_gear)
            bus.send(accel_msg)

            time.sleep(0.02) # 50Hz 주기 유지

    except can.CanError as e:
        print(f"\nCAN 에러 발생: {e}")
        print(f"'{can_interface}' 인터페이스가 활성화되어 있는지 확인해주세요.")
        print("예: sudo ip link set can0 up type can bitrate 500000")
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        global running
        running = False
        if bus:
            print("\n안전 정지를 위해 정지 명령(P기어, 속도 0)을 보냅니다.")
            # 기어는 P(0), 속도는 0으로 설정하여 안전하게 정지
            stop_msg = create_ad_accelerate_msg(speed_kph=0.0, gear=0)
            bus.send(stop_msg)
            time.sleep(0.1) # 메시지가 확실히 전송되도록 잠시 대기
            bus.shutdown()
            print("CAN bus 연결이 종료되었습니다.")

if __name__ == '__main__':
    main()

