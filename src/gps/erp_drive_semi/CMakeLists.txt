cmake_minimum_required(VERSION 3.0.2)
project(erp_drive_semi)

find_package(catkin REQUIRED COMPONENTS
  tf
  race
  rospy
  roscpp
  roslib
  roslint
  nav_msgs
  morai_msgs
  sensor_msgs
  geometry_msgs
  dynamic_reconfigure
)

find_package(fmt)

generate_dynamic_reconfigure_options(
  cfg/controller.cfg
)

add_service_files(
  FILES
  InitializePose.srv
  PermitParking.srv
)

add_message_files(
  FILES
  PathReferenceElement.msg
  PathReference.msg
)

# generate_messages의 주석을 푼다.
generate_messages(
  DEPENDENCIES
  geometry_msgs std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy sensor_msgs morai_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(
  control2
  src/controller.cpp
  src/pure_pursuit.cpp
  src/stanley_method.cpp
  src/controller_node.cpp
)

target_link_libraries(control2 ${catkin_LIBRARIES})
add_dependencies(control2 ${catkin_EXPORTED_TARGETS})
add_dependencies(control2 ${PROJECT_NAME}_gencfg)
add_dependencies(control2 ${PROJECT_NAME}_generate_messages_cpp)


add_executable(
  control_fmtc
  src/controller_fmtc.cpp
  src/pure_pursuit.cpp
  src/stanley_method.cpp
  src/controller_node.cpp
)

target_link_libraries(control_fmtc ${catkin_LIBRARIES})
add_dependencies(control_fmtc ${catkin_EXPORTED_TARGETS})
add_dependencies(control_fmtc ${PROJECT_NAME}_gencfg)
add_dependencies(control_fmtc ${PROJECT_NAME}_generate_messages_cpp)

# Specify where the Python scripts are located
catkin_install_python(PROGRAMS
  src/planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
