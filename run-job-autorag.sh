#!/bin/bash

set -xe

srun \
  --gpus=6 \
  --mem=144GB \
  --container-image=/data/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh \
  --container-workdir=`pwd` \
  --container-mounts=/data/nothnagel/rag-pipelines:/data/nothnagel/rag-pipelines \
  ./job_autorag.sh