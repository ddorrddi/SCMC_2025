cmake_minimum_required(VERSION 3.0.2)
project(lidar_object_detection)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
  message_generation
  laser_geometry
  dynamic_reconfigure
)

add_message_files(
   FILES
   Boundingbox.msg
   ObjectInfo.msg
   PointInfo.msg
   NearestObstacleInfo.msg
)


generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
)


generate_dynamic_reconfigure_options(
  cfg/config_tunnel_labacon.cfg
  cfg/config_tunnel_dynamic.cfg
  cfg/config_tunnel_static.cfg
  cfg/config_tunnel_semi.cfg
  cfg/config_delivery_info.cfg
  cfg/config_tunnel_big.cfg
)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES lidar_object_detection
  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs std_msgs geometry_msgs nav_msgs message_runtime laser_geometry dynamic_reconfigure
)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
add_executable(object_detection_tunnel_labacon src/object_detection_tunnel_labacon.cpp)
target_link_libraries(object_detection_tunnel_labacon ${catkin_LIBRARIES})
add_dependencies(object_detection_tunnel_labacon ${PROJECT_NAME}_gencfg)

add_executable(object_detection_delivery src/object_detection_delivery.cpp)
target_link_libraries(object_detection_delivery ${catkin_LIBRARIES})
add_dependencies(object_detection_delivery ${PROJECT_NAME}_gencfg)

add_executable(object_detection_tunnel_dynamic src/object_detection_tunnel_dynamic.cpp)
target_link_libraries(object_detection_tunnel_dynamic ${catkin_LIBRARIES})
add_dependencies(object_detection_tunnel_dynamic ${PROJECT_NAME}_gencfg)

add_executable(object_detection_tunnel_static src/object_detection_tunnel_static.cpp)
target_link_libraries(object_detection_tunnel_static ${catkin_LIBRARIES})
add_dependencies(object_detection_tunnel_static ${PROJECT_NAME}_gencfg)

add_executable(object_detection_tunnel_big src/object_detection_tunnel_big.cpp)
target_link_libraries(object_detection_tunnel_big ${catkin_LIBRARIES})
add_dependencies(object_detection_tunnel_big ${PROJECT_NAME}_gencfg)

add_executable(object_detection_tunnel_semi src/object_detection_tunnel_semi.cpp)
target_link_libraries(object_detection_tunnel_semi ${catkin_LIBRARIES})
add_dependencies(object_detection_tunnel_semi ${PROJECT_NAME}_gencfg)

