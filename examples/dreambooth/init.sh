#!/usr/bin/env bash -l

# INIT SCRIPT
# AWS LINUX UBUNTU g5.2xlarge
# DEEP LEARNING AMI PYTORCH 1.13.0

# Requires AWS USER VARIABLES (PROVISIONING)
: '
echo export NODE_ENV="**" | sudo tee -a /etc/profile
echo export ACCESS_KEY_ID="**" | sudo tee -a /etc/profile
echo export SECRET_ACCESS_KEY="**" | sudo tee -a /etc/profile
echo export HUGGINGFACE_TOKEN="**" | sudo tee -a /etc/profile
echo export GITHUB_ACCESS_TOKEN="**" | sudo tee -a /etc/profile
'

# CONDA (DIFFUSERS) ENV
# Here we create a single conda environment that
# stable-diffusion-webui and diffusers can both use
conda init bash 
eval "$(conda shell.bash hook)"
conda update -n base -c defaults conda -y
conda create --name diffusers python=3.10.6 -y
conda activate diffusers

# HUGGINGFACE
# Add hugging face and github credentials
sudo tee ${HOME}/.git-credentials <<EOF
https://hf_user:${HUGGINGFACE_TOKEN}@huggingface.co
https://${GITHUB_ACCESS_TOKEN}@github.com
EOF
sudo chmod 664 ${HOME}/.git-credentials
git config --global credential.helper store
##

# DOWNLOAD MODELS
# This adds the models we use for training locally, but can also be 
# retrieved from gpu-instance-s3fs. See launch.py in /diffusers/examples/dreambooth
sudo apt-get install git-lfs
git lfs install
mkdir default-models
cd default-models
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
cd stable-diffusion-v1-5
git lfs pull
rm -rf .git
cd ..
git clone https://huggingface.co/stabilityai/sd-vae-ft-mse
cd sd-vae-ft-mse
git lfs pull
rm -rf .git

# NODE API
# Install Node, pm2 and run API
cd ${HOME}
sudo apt-get install apt-transport-https curl software-properties-common -y
sudo curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install pm2 -g
git clone https://${GITHUB_ACCESS_TOKEN}@github.com/BuffMcBigHuge/gpu-instance-api.git
cd gpu-instance-api
npm install

# S3FS
# Save S3FS AWS Credentials
# Ensure ACCESS_KEY_ID and SECRET_ACCESS_KEY are stored as env variables (AWS User data)
cd ${HOME}
file=".passwd-s3fs"
echo "$ACCESS_KEY_ID:$SECRET_ACCESS_KEY" > $file
cat $file
chmod 600 .passwd-s3fs
sudo apt install s3fs -y
mkdir gpu-instance-s3fs
s3fs gpu-instance-s3fs ${HOME}/gpu-instance-s3fs -o passwd_file=${HOME}/.passwd-s3fs

# REBOOT SCRIPTS
# These commands will run as ubuntu user when the machine is restarted
# 1) Start S3SF, mounts to gpu-instance-s3sf
# 2) Start pm2 node API on 3000
# 3) Start stable-diffusion-webui on port 7860
cd ${HOME}
sudo tee /etc/rc.local <<EOF
#!/bin/bash
su ubuntu -c 's3fs gpu-instance-s3fs ${HOME}/gpu-instance-s3fs -o passwd_file=${HOME}/.passwd-s3fs'
su ubuntu -c 'cd ${HOME}/gpu-instance-api && pm2 start ecosystem.config.js --env ${NODE_ENV}'
'
EOF
# We will not run the webui since we're using inference.py in diffusers
# su ubuntu -c 'cd ${HOME}/stable-diffusion-webui && conda run -n diffusers --no-capture-output python launch.py --ckpt-dir ../gpu-instance-s3fs/models --api --listen --xformers'
sudo chmod +x /etc/rc.local
##

# DEPENDENCIES
# These are the pip dependices for launch.py in /stable-diffusion-webui and /diffusers/examples/dreambooth
pip install torchvision==0.13.1 torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install accelerate==0.12.0 transformers ninja bitsandbytes
pip install deepspeed pyheif piexif python-resize-image
pip install -U --pre triton
pip install git+https://github.com/facebookresearch/xformers.git@103e863db94f712a96c34fc8e78cfd58a40adeee

# DREAMBOOTH
# Installs ShivamShrirao/diffusers fork
# This fork includes modifications to launch.sh
cd ${HOME}
git clone https://github.com/BuffMcBigHuge/diffusers.git
cd diffusers
pip install git+https://github.com/BuffMcBigHuge/diffusers.git
sudo chmod +x ${HOME}/diffusers/examples/dreambooth/launch.sh

# INFERENCE
pip install gfpgan basicsr realesrgan

# AUTOMATIC1111 WEBUI API
# Installs automatic1111/stable-diffusion-webui fork
# This fork includes modifications to API override_settings (model selector)
# and also on dependices to work well with dreambooth (transformers, diffusion)
# cd ${HOME}
# git clone https://github.com/BuffMcBigHuge/stable-diffusion-webui.git
# cd stable-diffusion-webui
# pip install -r requirements.txt

# UPGRADE
# Not sure if this is nessary anymore. SD 2.0 requirement but untest atm
pip install --upgrade transformers==4.21.0 diffusers==0.7.2 accelerate scipy ftfy

# ACCELERATE CONFIG
# Auto configures accelerate instead of `accelerate config`
cd ${HOME}
mkdir -p  ${HOME}/.cache/huggingface/accelerate
sudo tee ${HOME}/.cache/huggingface/accelerate/default_config.yaml <<EOF
command_file: null
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
gpu_ids: all
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false
EOF
sudo chmod 664 ${HOME}/.cache/huggingface/accelerate/default_config.yaml

# EXAMPLE LAUCH SCRIPTS
# These are example launch scripts to test integration
: '
# DREAMBOOTH
cd /home/ubuntu/diffusers/examples/dreambooth && conda run -n diffusers --no-capture-output ./launch.sh -u "Lz4dhJQWfeQ0rDHntZnbAyXnLe62" -c "man" -m "ModelName"
# WEBUI API
cd /home/ubuntu/stable-diffusion-webui && conda run -n diffusers --no-capture-output python launch.py --ckpt-dir ../gpu-instance-s3fs/models --api --listen --xformers
'

conda deactivate

# REBOOT
sudo reboot

# EOF

# OLD METHODS

# pip install git+https://github.com/facebookresearch/xformers@1d31a3a#egg=xformers

# PYTHON
# sudo add-apt-repository ppa:deadsnakes/ppa -y
# sudo apt update
# sudo apt upgrade -y
# sudo apt install python3.10 python3-pip python3-venv python-is-python3 python3.10-distutils -y
# curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
# sudo ln -sf /usr/bin/pip3 /usr/bin/pip
# sudo ln -sf /usr/bin/python3.10 /usr/bin/python3

# CONDA
# cd ${HOME}
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
# bash ~/miniconda.sh -b -p $HOME/miniconda
# export PATH=${HOME}/miniconda/bin:$PATH
# conda init zsh

# DRIVERS
# sudo apt-get install nvidia-driver-510 nvidia-utils-510 -y