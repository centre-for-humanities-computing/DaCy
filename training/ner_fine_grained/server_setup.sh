# Setup for initial training of models for Virtual Machine
# System: image based on Cuda + Jupyter on Ubuntu v. 20.04
# GPU: uc-t4-1 / uc-t4gpu
# Access: UCloud, Aalborg University 


# upgrade system
sudo apt full-upgrade -y
sudo apt update -y
# nvidia drivers
sudo apt install nvidia-headless-460 nvidia-utils-460 -y
# general util
sudo apt install unzip git-lfs -y
# python util
sudo apt install python3-pip python3.8-venv -y
# sudo reboot to make the changes work
sudo reboot