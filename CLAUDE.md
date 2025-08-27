# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 코드 작업을 할 때 가이드를 제공합니다.

## 빌드 및 개발 명령어

### 프로젝트 빌드
```bash
# 빌드 오류 시 ROS 의존성 설치
rosdep install --from-paths src --ignore-src -r

# 전체 워크스페이스 빌드
catkin_make

# 빌드 후 워크스페이스 소스
source devel/setup.bash
```

### 일반적인 개발 작업
- 프로젝트는 catkin 빌드 시스템을 사용하는 ROS(Robot Operating System)를 사용
- 빌드 출력은 `build/` 및 `devel/` 디렉토리로 이동
- 빌드 후 `devel/setup.bash`를 소스하여 ROS 환경 설정

## 아키텍처 개요

SCMC 2025 경진대회를 위한 자율주행 차량 소프트웨어 스택으로, ROS 기반으로 구축되었습니다. 시스템은 자율주행 시나리오를 위해 여러 센서 모달리티와 제어 알고리즘을 통합합니다.

### 핵심 시스템 컴포넌트

**GPS/내비게이션 (`src/gps/`)**
- `erp_drive`: 자율주행을 위한 메인 제어 및 경로 계획 패키지
- `erp_drive_semi`: 반자율 주행 변형  
- `ublox_f9p`: GPS 하드웨어 드라이버 및 설정
- `handsfree_ros_imu`: IMU 센서 통합
- `race`: 차량 제어를 위한 시리얼 통신

**카메라 비전 (`src/camera/`)**
- `lanenet`: 딥러닝 모델을 사용한 차선 감지
- `traffic_light`: YOLO를 사용한 신호등 감지
- `delivery`: 배송 표지판 감지
- `bev`: 카메라 피드를 위한 조감도 변환

**LiDAR 처리 (`src/lidar/`)**
- `lidar_object_detection`: 포인트 클라우드 처리 및 장애물 감지
- `waypoint_maker`: 경로 계획 유틸리티

**외부 의존성**
- `MORAI-ROS_morai_msgs`: MORAI 시뮬레이터 통합을 위한 메시지 정의
- `rtcm_msgs`: GPS용 RTCM 메시지 지원

### 주요 런치 파일 및 사용법

**메인 제어 런치 파일:**
- `src/gps/erp_drive/launch/control.launch` - 기본 자율주행 런치
- `src/gps/erp_drive/launch/control_p1.launch` - 1단계 경진대회 설정  
- `src/gps/erp_drive/launch/control_p2.launch` - 2단계 경진대회 설정
- `src/gps/erp_drive/launch/control_p3.launch` - 3단계 경진대회 설정

**비전 시스템 런치 파일:**
- `src/camera/lanenet/launch/lanenet_drive_morai.launch` - MORAI 시뮬레이터용 차선 감지
- `src/camera/traffic_light/launch/yolo_traffic_light_morai.launch` - 신호등 감지

**센서 런치 파일:**
- `src/gps/ublox_f9p/ublox_gps/launch/ublox_zed-f9p_rover.launch` - GPS 로버 설정
- `src/gps/handsfree_ros_imu/launch/handsfree_imu.launch` - IMU 센서 런치

### 경로 계획 및 제어

시스템은 `src/gps/erp_drive/paths/`에 저장된 미리 정의된 경로 파일을 사용:
- `morai.txt` - MORAI 시뮬레이터 경로
- `kcity_*.txt` - K-City 테스트 트랙 경로  
- `parking.txt` - 주차 시나리오 경로
- `tunnel.txt` - 터널 주행 경로

**제어 아키텍처:**
- C++로 구현된 Pure pursuit 제어기
- Python 기반 경로 계획 및 장애물 회피
- 센서 데이터 통합을 위한 융합 모듈
- 실시간 파라미터 조정을 위한 동적 재설정

### 메시지 타입 및 통신

**커스텀 메시지:**
- 경로 계획: `PathReference.msg`, `PathReferenceElement.msg`
- 객체 감지: `Boundingbox.msg`, `NearestObstacleInfo.msg`, `ObjectInfo.msg`
- 차량 제어: ERP42 플랫폼용 다양한 제어 명령 메시지
- 센서 융합: `IoUFusionMsg.msg`

### 개발 워크플로

1. **환경 설정**: `source devel/setup.bash`로 ROS 워크스페이스 소스
2. **빌드**: 패키지 빌드에 `catkin_make` 사용
3. **의존성**: 의존성이 누락된 경우 `rosdep install` 실행  
4. **테스트**: 시각화를 위해 다양한 `rviz/` 디렉토리의 RViz 설정 사용
5. **설정**: 각 패키지의 `.cfg` 파일을 통해 파라미터 조정 가능

### 하드웨어 통합

시스템은 다음을 위해 설계됨:
- ERP42 자율주행 차량 플랫폼
- MORAI 시뮬레이션 환경 호환성
- u-blox F9P를 사용한 GPS RTK 포지셔닝
- LiDAR 센서 통합
- 카메라 기반 인식 시스템
- 차량 상태 추정을 위한 IMU

### 경진대회 모드

소프트웨어는 여러 경진대회 단계 지원:
- **1단계**: 기본 자율주행
- **2단계**: 향상된 인식 및 장애물 회피  
- **3단계**: 배송 작업을 포함한 복잡한 시나리오
- **반자율 모드**: 운전자 보조 기능