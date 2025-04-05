#!/usr/bin/env bash

PROJECT_DIR=$(dirname "$(dirname "$(realpath "$0")")")

# checkpoints, wandb, and other output files will be stored alongside the project directory
PROJECT_PARENT=$(dirname "$PROJECT_DIR")

set -x

export NCCL_CUMEM_ENABLE=0
export WANDB_MODE=online
export WANDB_DIR=$PROJECT_PARENT
export OMP_NUM_THREADS=8


BS=256
EP=5
LR=1e-5

TRIAL_NAME=sft_longcot
DATE_SUFFIX=$(date +"%Y%m%d_%H%M%S")
MODEL_PATH=Qwen/Qwen2.5-Math-1.5B
SAVE_PATH=$PROJECT_PARENT/ckpts/longcot_sft_$DATE_SUFFIX
DATA_PATH=$PROJECT_DIR/data/train/math3k_longcot.jsonl

read -r -d '' training_commands <<EOF
src.cli.train_sft \
   --max_len 16384 \
   --dataset $DATA_PATH \
   --input_key prompt \
   --output_key solution \
   --train_batch_size $BS \
   --micro_train_batch_size 1 \
   --apply_chat_template \
   --max_samples 50000000 \
   --pretrain $MODEL_PATH \
   --save_path $SAVE_PATH \
   --ckpt_path $SAVE_PATH \
   --disable_ds_ckpt \
   --max_ckpt_num 100 \
   --save_hf_ckpt \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs $EP \
   --bf16 \
   --flash_attn \
   --learning_rate $LR \
   --lr_scheduler cosine_with_min_lr \
   --gradient_checkpointing \
   --packing_samples \
   --use_wandb $WANDB_API_KEY \
   --wandb_project sjtu_cs2916_baseline \
   --wandb_group sft \
   --wandb_run_name $TRIAL_NAME
EOF

torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 \
    --master_addr "127.0.0.1" --master_port 12345 -m ${training_commands}
