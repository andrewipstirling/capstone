# aruco_box

This repository provides a cube for Gazebo. An ArUco marker is attached to each side of the cube.

![gazebo](doc/gazebo.png)

The (edited) parameters are:
- marker width is 10% of the cube width
- ArUco marker IDs are 0,1,..,5
- Cube size is 0.04x0.04x0.04m

## Installation

Download the repository and add this directory to the **GAZEBO_MODEL_PATH** environment variable. To add to *.bashrc* use 
```
cd ~
echo "export GAZEBO_MODEL_PATH=$GAZEBO_RESOURCE_PATH:~/path_to_captsone/capstone/aruco_box/" >> .bashrc
```

## Usage

Open Gazebo and navigate to the "Insert" tab. There, you will find the "Box with ArUco markers" model. 

## Customization

You can create the texture for the cube yourself. See the script "src/create_marker_tile_image.py" for more details. 
The Blender model is also added to the repo (aruco_box/aruco_marker.blend). 

