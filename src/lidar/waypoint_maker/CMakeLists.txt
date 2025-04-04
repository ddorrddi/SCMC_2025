cmake_minimum_required(VERSION 3.0.2)
project(waypoint_maker)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## C++14 표준으로 컴파일 설정
add_compile_options(-std=c++14)

## Catkin 매크로와 라이브러리 찾기
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  pcl_conversions
  pcl_ros
  message_generation
  dynamic_reconfigure
  lidar_trafficcone_detection
)

## PCL 라이브러리 찾기
find_package(PCL 1.10 REQUIRED)

## 메시지 파일 추가
add_message_files(
  FILES
  Waypoint.msg
)

## 메시지 생성
generate_messages(
  DEPENDENCIES
  std_msgs
)

generate_dynamic_reconfigure_options(
    cfg/waypoint_maker.cfg
)

## Catkin 패키지 선언
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs pcl_conversions pcl_ros message_runtime dynamic_reconfigure lidar_trafficcone_detection
  DEPENDS PCL
)

## 추가 헤더 파일 위치 지정
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_NAME}
)

## 라이브러리 디렉토리 지정
link_directories(${PCL_LIBRARY_DIRS})

## 정의 추가
add_definitions(${PCL_DEFINITIONS})