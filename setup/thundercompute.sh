#!/bin/bash

echo "================== APT Update =================="
sudo apt update
echo "================== Install pip =================="
sudo apt install -y python3-pip
echo "================== Make virtual env =================="
python3 -m venv venv
source venv/bin/activate
echo "================== Install pip packages =================="
pip install -U torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
pip install -U matplotlib numpy datasets


echo "================== Make jupyter kernel =================="
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "venvkernel"
