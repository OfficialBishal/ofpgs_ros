cmake_minimum_required(VERSION 3.0.2)
project(ofpgs_ros)

## Compile as C++11
add_compile_options(-std=c++11)

## Find catkin and required ROS packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  image_transport
  message_filters
  tf2_ros
  tf2_geometry_msgs
)

find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS 
    rospy roscpp std_msgs sensor_msgs geometry_msgs visualization_msgs
    cv_bridge image_transport message_filters tf2_ros tf2_geometry_msgs
)

###########
## Build ##
###########
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

#############
## Install ##
#############

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/foundationpose_pose_estimation_node.py
  scripts/sam_segmentation_node.py
  scripts/grounded_sam_segmentation_node.py
  scripts/circle_table_mover.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

## Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

## Install mesh files
install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
)

## Install setup scripts
install(DIRECTORY setup/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/setup
  USE_SOURCE_PERMISSIONS
)

## Install shell scripts
install(PROGRAMS
  scripts/run_foundationpose.sh
  scripts/run_sam_segmentation.sh
  scripts/run_grounded_sam_segmentation.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
