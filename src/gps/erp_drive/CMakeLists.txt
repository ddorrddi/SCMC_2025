cmake_minimum_required(VERSION 3.0.2)
project(erp_drive)

find_package(catkin REQUIRED COMPONENTS
  tf
  race
  rospy
  roscpp
  roslib
  roslint

  # fusion용
  std_msgs
  cv_bridge
  message_filters

  #

  nav_msgs
  morai_msgs
  sensor_msgs
  geometry_msgs
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/controller.cfg
  cfg/FusionExtrinsics.cfg
  cfg/CaliReal.cfg
)

add_service_files(
  FILES
  InitializePose.srv
)

add_message_files(
  FILES
  PathReferenceElement.msg
  PathReference.msg
  IoUFusionMsg.msg
)

# generate_messages의 주석을 푼다.
generate_messages(
  DEPENDENCIES
  geometry_msgs std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy sensor_msgs morai_msgs message_runtime std_msgs message_filters
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(
  control
  src/controller.cpp
  src/pure_pursuit.cpp
  src/controller_node.cpp
)

target_link_libraries(control ${catkin_LIBRARIES})
add_dependencies(control ${catkin_EXPORTED_TARGETS})
add_dependencies(control ${PROJECT_NAME}_gencfg)
add_dependencies(control ${PROJECT_NAME}_generate_messages_cpp)
