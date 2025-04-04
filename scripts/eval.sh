#!/usr/bin/env bash

export HOST_IP=$MASTER_ADDR
export NCCL_CUMEM_ENABLE=0

ckpt_path=$1
python src/evaluation/evaluation.py --model_path $ckpt_path --max_tokens 16384