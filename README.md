# Capstone
## Dependencies
Ubuntu 20.04 + 
ROS Noetic: http://wiki.ros.org/noetic/Installation/Ubuntu
catkin_tools: https://catkin-tools.readthedocs.io/en/latest/installing.html#installing-on-ubuntu-with-apt-get
## Installation Steps

### Git clone capstone
```
# Using https
git clone https://github.com/andrewipstirling/capstone.git
# store path for later
tmp_path=$(pwd)
```

### Create catkin_ws
```
mkdir capstone_ws
cd capstone_ws
mkdir src
catkin init
ln -s $tmp_var/capstone/ros_aruco_gazebo src
```

## Running Empty Simulation
```
catkin build
# re-source ws
source devel/setup.bash
roslaunch ros_aruco_gazebo gazebo_sim.launch
```
