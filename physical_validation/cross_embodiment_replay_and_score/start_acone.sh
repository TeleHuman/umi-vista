#!/bin/bash

source /opt/ros/humble/setup.bash
source ~/ARX_X5/ROS2/X5_ws/install/setup.bash

cd ~/ARX_X5/00-sh/ROS2/AC_one
./04joint_control.sh

cd ~/rm75_TeleAI
export ROBOT_NAME="acone"