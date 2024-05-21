#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2 
	--pipeline-model-parallel-size 1 
)

BERT_MODEL_ARGS=(
    --num-layers 1 
    --hidden-size 512 
    --num-attention-heads 8 
    --seq-length 1 
    --max-position-embeddings 512
)

TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 32 
    --train-iters 1000000 
    --weight-decay 1e-2 
    --clip-grad 1.0 
    --fp16
    --lr 0.0001
    --lr-decay-iters 990000 
    --lr-decay-style linear 
    --min-lr 1.0e-5 
    --weight-decay 1e-2 
    --lr-warmup-fraction .01 
    --clip-grad 1.0 
    --use-mcore-models
)

torchrun ${DISTRIBUTED_ARGS[@]} training.py ${MODEL_PARALLEL_ARGS[@]} ${TRAINING_ARGS[@]} ${BERT_MODEL_ARGS[@]} 