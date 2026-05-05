MAJOR=$(grep nvidia-caps-imex-channels /proc/devices | awk '{print $1}')

# Use mknod to open fabric channel 0
sudo mkdir -p /dev/nvidia-caps-imex-channels/
sudo mknod /dev/nvidia-caps-imex-channels/channel0 c $MAJOR 0
sudo chmod 666 /dev/nvidia-caps-imex-channels/channel0

# Optional: set environment variables to use VMM and distributed
export TILESCALE_USE_VMM=1
export TILESCALE_USE_DISTRIBUTED=1
