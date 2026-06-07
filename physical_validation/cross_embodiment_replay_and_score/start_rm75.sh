# conda activate rm7500_env
# sudo ip addr add 192.168.31.110/24 dev enp0s31f6   
# sudo chmod 777 /dev/ttyACM*
sudo chmod 777 /dev/ttyUSB*
rm -f /dev/shm/D405*
rm -f /dev/shm/L515*
rm -f /dev/shm/rm75*

export ROBOT_NAME="rm75"