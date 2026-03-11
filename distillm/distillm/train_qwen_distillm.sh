#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5 distillation (teacher 9B -> student 0.8B) with DistiLLM-style online distillation.
# Fill the paths below, then run:
#   bash scripts/qwen3_5/distillm/train_0p8B_9B_distillm.sh ${DISTILLM_PATH} ${MASTER_PORT} ${GPU_NUM}

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
                  --nnodes ${NNODES} \
                  --node_rank ${NODE_RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

# project
BASE_PATH=${1-"/home/caonan/lyl/edge_model/distillm/distillm"}

# model (can be local folders or HF repo ids)
STUDENT_CKPT_NAME="Qwen3.5-0.8B"
STUDENT_CKPT="/home/caonan/lyl/models/${STUDENT_CKPT_NAME}"

TEACHER_CKPT_NAME="Qwen3.5-9B"
TEACHER_CKPT="/home/caonan/lyl/models/${TEACHER_CKPT_NAME}"

# data (processed bin/idx directory; must contain train_0.bin/.idx and valid_0.bin/.idx)
DATA_DIR="/home/caonan/lyl/edge_model/distillm/distillm/data/processed/ai_data_qwen/qwen/"

# hp (safe start for 4x32GB; adjust as needed)
MAX_LENGTH=1024
MAX_PROMPT_LENGTH=512
BATCH_SIZE=1
GRAD_ACC=16
EVAL_BATCH_SIZE=1
LR=5e-5
EPOCHS=3
KD_RATIO=1.0

# runtime
SAVE_PATH="${BASE_PATH}/results/qwen3_5/train/distillm/0p8B_9B"
SEED=10

# DistiLLM knobs
TYPE="adaptive-srkl"
INIT_THRESHOLD=0.0
LOSS_EPS=0.1
CAPACITY=2000

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${STUDENT_CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${STUDENT_CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen"
OPTS+=" --gradient-checkpointing"

# (Optional) Use LoRA to cut optimizer-state memory a lot.
# Uncomment these if you still hit OOM on full fine-tuning.
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r 8"
OPTS+=" --peft-lora-alpha 16"
OPTS+=" --peft-lora-dropout 0.05"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"

# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio ${KD_RATIO}"

# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"

# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --save-interval 100"
OPTS+=" --eval-interval 100"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"

# seed
OPTS+=" --seed ${SEED}"

# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2.json"

# distill type
OPTS+=" --type ${TYPE}"

# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"

# DistiLLM (student generation + replay)
OPTS+=" --student-gen"
OPTS+=" --init-threshold ${INIT_THRESHOLD}"
OPTS+=" --loss-eps ${LOSS_EPS}"
OPTS+=" --capacity ${CAPACITY}"

export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${BASE_PATH}

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo "${CMD}"
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p "${SAVE_PATH}"

${CMD}

