#!/bin/bash

# show output in terminal
set -xe
# install python packages
pip install -r requirements_autorag.txt
pip uninstall transformer-engine
pip install transformers
# install git lfs
apt-get update
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install
# run training script unbuffered
python -u ./autorag/ragas_llama3_gpu.py 