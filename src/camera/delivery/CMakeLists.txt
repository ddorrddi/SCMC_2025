cmake_minimum_required(VERSION 3.0.2)
project(delivery_yolo)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  vision_msgs
  message_generation
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime roscpp rospy std_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
