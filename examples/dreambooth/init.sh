#!/usr/bin/env bash -l

# Init Script for AWS LINUX UBUNTU g5.2xlarge

# Requires AWS USER VARIABLES (PROVISIONING)
: '
echo export ACCESS_KEY_ID="**" | sudo tee -a /etc/profile
echo export SECRET_ACCESS_KEY="**" | sudo tee -a /etc/profile
echo export HUGGINGFACE_TOKEN="**" | sudo tee -a /etc/profile
echo export GITHUB_ACCESS_TOKEN="**" | sudo tee -a /etc/profile
'

# PYTHON
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update && sudo apt upgrade -y
sudo apt install python3.10 python3-pip python3-venv python-is-python3 -y

# CONDA
cd ${HOME}
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH=${HOME}/miniconda/bin:$PATH
conda init zsh
conda init bash 
eval "$(conda shell.bash hook)"
conda update -n base -c defaults conda -y

# HUGGINGFACE
# Ensure HUGGINGFACE_TOKEN is stored as env variables (AWS User data)
sudo tee ${HOME}/.git-credentials <<EOF
https://hf_user:${HUGGINGFACE_TOKEN}@huggingface.co
https://${GITHUB_ACCESS_TOKEN}@github.com
EOF
chmod 664 ${HOME}/.git-credentials
##

# MODELS
# TODO: Store in S3FS?
: '
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
cd ${HOME}
'

# API
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
cd ${HOME}
sudo tee /etc/rc.local <<EOF
#!/bin/bash
su ubuntu -c '\
s3fs gpu-instance-s3fs ${HOME}/gpu-instance-s3fs -o passwd_file=${HOME}/.passwd-s3fs && \
cd ${HOME}/gpu-instance-api && pm2 start ecosystem.config.js --env production && \
export PATH=${HOME}/miniconda/bin:$PATH && \
cd ${HOME}/stable-diffusion-webui && conda run --no-capture-output -n webui ./webui.sh'
EOF
sudo chmod +x /etc/rc.local
##

# XFORMERS
cd ${HOME}
conda create --name diffusers python=3.10.6 -y
conda activate diffusers
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
pip install -U --pre triton
conda install xformers -c xformers/label/dev -y
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install functorch==0.2.1 ninja bitsandbytes
# -> Once installed, set the flag _is_functorch_available = True in xformers/__init__.py
sed -i 's/_is_functorch_available: bool = False/_is_functorch_available: bool = True/g' xformers/__init__.py
# pip install -e .
pip install --verbose --no-deps -e .
# python setup.py clean && python setup.py develop

# DIFFUSERS
cd ${HOME}
git clone https://github.com/BuffMcBigHuge/diffusers.git
cd diffusers/examples/dreambooth
pip install git+https://github.com/BuffMcBigHuge/diffusers.git
pip install -r requirements.txt
pip install deepspeed pyheif piexif python-resize-image
sudo chmod +x launch.sh
conda deactivate

# huggingface-cli login
# accelerate config
cd ${HOME}
mkdir -p  ${HOME}/.cache/huggingface/accelerate
sudo tee ${HOME}/.cache/huggingface/accelerate/default_config.yml <<EOF
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
use_cpu: false
EOF
sudo chmod 664 ${HOME}/.cache/huggingface/accelerate/default_config.yml

# AUTOMATIC1111
cd ${HOME}
git clone https://github.com/BuffMcBigHuge/stable-diffusion-webui.git
cd stable-diffusion-webui
conda create --name webui python=3.10.6 -y
conda activate webui
pip install -r requirements.txt
# pip install torchvision==0.13.1
# pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Update webui-user
sudo tee -a webui-user.sh <<EOF

export COMMANDLINE_ARGS="--ckpt-dir ../gpu-instance-s3fs/models --api --listen --xformers"
EOF

# Activate VENV environment and manually install xformers again
# webui.sh script doesn't install xformers successfully
python -m venv venv
source venv/bin/activate
cd ${HOME}/xformers
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
pip install git+https://github.com/facebookresearch/xformers.git@v0.0.13#egg=xformers
pip install -U --pre triton
pip install --verbose --no-deps -e .
pip install -r requirements.txt
pip install functorch==0.2.1 ninja bitsandbytes
deactivate

conda deactivate
# bash webui.sh

# DRIVERS
# sudo apt-get install nvidia-driver-510 nvidia-utils-510 -y

# REBOOT
sudo reboot