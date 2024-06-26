cmake_minimum_required(VERSION 3.10)
project(trajectory_generation)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)


## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
    rospy
    std_msgs
    sensor_msgs
    gazebo_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS       
  LIBRARIES
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    gazebo_msgs
  DEPENDS 
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
	include
	${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS 
  src/follow_trajectory.py
  src/control_pointer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)