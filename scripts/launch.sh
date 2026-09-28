#!/bin/bash

export DATA_DIR
export CKPT_DIR
export NSIGHT_LOG_DIR
export NSIGHT_FILE_NAME
export MODE
export LOCAL_GPU_IDS
export NUM_GPUS

###########################################################################
# Problem 0: Generate Nsight log
# Find the correct way to generate Nsight log.
###########################################################################

#   --trace              : CUDA API/kernels, NVTX ranges (Epoch/Batch/forward/...), OS runtime, cuDNN, cuBLAS
#   --cuda-memory-usage  : GPU memory usage timeline
#   --sample/--cpuctxsw  : Vast.ai containers are unprivileged, so CPU sampling is disabled
#   NSYS_EXTRA_ARGS      : optional extra options, e.g. NSYS_EXTRA_ARGS="--gpu-metrics-devices=all"
# DDP children created by mp.spawn are traced into the same report.
CUDA_VISIBLE_DEVICES=$LOCAL_GPU_IDS nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="$NSIGHT_LOG_DIR/$NSIGHT_FILE_NAME" \
    $NSYS_EXTRA_ARGS \
    python train_cifar.py \
    --num_gpu=$NUM_GPUS \
    --data="$DATA_DIR" \
    --ckpt="$CKPT_DIR" \
    --mode="$MODE" \
    --save_ckpt