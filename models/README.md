# models

This repository provides various models used for the simulation

![gazebo](doc/gazebo.png)


## Installation

Download the repository and add this directory to the **GAZEBO_MODEL_PATH** environment variable. To add to *.bashrc* use 
```
cd ~
echo "export GAZEBO_MODEL_PATH=$GAZEBO_RESOURCE_PATH:~/path_to_captsone/capstone/models/" >> .bashrc
```

## Usage

Open Gazebo and navigate to the "Insert" tab. There, you will find the models described in this folder. 

## Customization

You can create new markers with associated IDs using the *gen_marker_png.py* script. First copy a previous aruco_marker_x folder, and change the naming of relevant files. Then run the script changing the variable name ID to the required marker ID of the aruco6x6_250 dictionary.


