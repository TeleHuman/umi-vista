#!/bin/zsh

echo current python: $(which python)
echo current accelerate: $(which accelerate)

# Dataset configuration
DATASET_REPO_ID="IPEC-COMMUNITY/bridge_orig_lerobot"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"

# Pretrained model configuration  
### the pi05_base model must be the newest version uploaded to HF hub in LeRobot collection
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi05_base"

### accelerate launch arguments
GPUS=2
MAIN_PROCESS_PORT=29500
MIXED_PRECISION="no"
GRADIENT_ACCUMULATION_STEPS=1

### Lerobot Training Parameters
BATCH_SIZE=32
TOTAL_STEPS=30000
SAVE_FREQ=10000
LEARNING_RATE=0.000025
ACTION_CHUNK_SIZE=4
NUM_WORKERS=12
SEED=42
USE_TENSORBOARD=true

# ------------------------------------------------------------
### generate output directory
DATE1=$(date "+%y-%m-%d")
TIME1=$(date "+%H-%M-%S")

MODEL_NAME="pi0"
DATASET_NAME="${DATASET_REPO_ID##*/}"
GPU_NUM="gpu${GPUS}"
BS="bs${BATCH_SIZE}"
STEPS_K=$(($TOTAL_STEPS / 1000))
STEPS="s${STEPS_K}K"
LR_SCI=$(printf "%.0e" $LEARNING_RATE | sed 's/e-0/e-/')
LR="lr${LR_SCI}"
CHUNK_SIZE="ck${ACTION_CHUNK_SIZE}"
SEED_STR="seed${SEED}"

MY_HOME="/gemini/platform/public/embodiedAI/users/ysy/data"
OUTPUT_DIR="$MY_HOME/train_pi0/${DATASET_NAME}/${DATE1}_${TIME1}_${MODEL_NAME}_${GPU_NUM}_${CHUNK_SIZE}_${LR}_${BS}_${STEPS}_${SEED_STR}"
echo "Output dir: $OUTPUT_DIR"
# ------------------------------------------------------------

cd /gemini/space/users/ysy/project/RhodesLeRobot_new_in/
### accelerate launch command
accelerate launch \
    --num_processes=$GPUS \
    --config_file=training_scripts/accelerate_configs/accelerate_default.yaml \
    --main_process_port=$MAIN_PROCESS_PORT \
    --mixed_precision=$MIXED_PRECISION \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    src/lerobot/scripts/lerobot_train_o.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.image_transforms.enable=true \
    --dataset.wrist_transforms.enable=true \
    --policy.path=$POLICY_PATH \
    --policy.local_files_only=true \
    --policy_optimizer_lr=$LEARNING_RATE \
    --output_dir=$OUTPUT_DIR --use_tensorboard=$USE_TENSORBOARD \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --seed=$SEED \
    --policy.chunk_size=$ACTION_CHUNK_SIZE --policy.n_action_steps=$ACTION_CHUNK_SIZE