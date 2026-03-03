#!/bin/bash

sudo apt update
sudo apt install python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -U torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
pip install -U matplotlib numpy datasets
