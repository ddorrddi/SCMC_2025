cmake_minimum_required(VERSION 3.0.2)
project(lidar_trafficcone_detection)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  pcl_conversions
  
  roscpp
  nav_msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  dynamic_reconfigure
)

find_package(PCL 1.8 REQUIRED)

add_message_files(
    FILES
    Delivery.msg
    ObjectInfo.msg
    Boundingbox.msg
    Trafficcone.msg
    DriveValues.msg
    DynamicVelocity.msg
)
    
generate_messages(
    DEPENDENCIES
    nav_msgs
    std_msgs
    sensor_msgs 
    geometry_msgs
)

generate_dynamic_reconfigure_options(
    cfg/traffic_hyper_parameter.cfg
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs visualization_msgs nav_msgs geometry_msgs std_msgs dynamic_reconfigure message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} src/traffic_cone.cpp)

add_executable(traffic_cone src/traffic_cone.cpp)
target_link_libraries(traffic_cone ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(traffic_cone ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(traffic_cone ${${PROJECT_NAME}_EXPORTED_TARGETS})
add_dependencies(traffic_cone ${catkin_EXPORTED_TARGETS})
add_dependencies(traffic_cone ${PROJECT_NAME}_gencfg)