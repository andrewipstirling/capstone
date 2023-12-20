# Capstone

## Installation Steps

### Git clone capston
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

## Running
```
catkin build
# re-source ws
source devel/setup.bash
roslaunch ros_aruco_gazebo gazebo_sim.launch
```
