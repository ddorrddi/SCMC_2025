cmake_minimum_required(VERSION 3.0.2)
project(static_obstacle_detection)

find_package(catkin REQUIRED COMPONENTS
    pcl_conversions
    pcl_ros
    roscpp
    sensor_msgs
    visualization_msgs
    nav_msgs
    geometry_msgs
    std_msgs
    message_generation
    dynamic_reconfigure
)

find_package(PCL 1.8 REQUIRED)

add_message_files(
  FILES
  Boundingbox.msg
  ObstacleInfo.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
)

generate_dynamic_reconfigure_options(
    cfg/static_hyper_parameter.cfg
    cfg/delivery_hyper_parameter.cfg
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES static_obstacle_detection
 CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs visualization_msgs nav_msgs geometry_msgs std_msgs dynamic_reconfigure
 DEPENDS system_lib
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CATKIN_DEVEL_PREFIX}/include
)

add_executable(static_obstacle_detection src/static_obstacle.cpp)
add_dependencies(static_obstacle_detection ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(static_obstacle_detection ${PROJECT_NAME}_gencfg)
target_link_libraries(static_obstacle_detection
  ${catkin_LIBRARIES} ${PCL_LIBRARIES}
)

add_executable(delivery_obstacle_detection src/delivery_obstacle.cpp)
add_dependencies(delivery_obstacle_detection ${PROJECT_NAME}_generate_messages_cpp)
add_dependencies(delivery_obstacle_detection ${PROJECT_NAME}_gencfg)
target_link_libraries(delivery_obstacle_detection
  ${catkin_LIBRARIES} ${PCL_LIBRARIES}
)
