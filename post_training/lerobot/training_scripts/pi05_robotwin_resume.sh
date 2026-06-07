#!/bin/bash

echo current python: $(which python)
echo current accelerate: $(which accelerate)

gpu_id=0
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Dataset configuration
# DATASET_REPO_ID="HuggingFaceVLA/libero"

DATASET_REPO_ID="RoboTwin2/v30/demo_clean_50tasks"
# DATASET_REPO_ID="IPEC-COMMUNITY/bridge_orig_lerobot"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"
DATASET_REVISION="v3.0"
# DATASET_REVISION=null

# Pretrained model configuration  
### the pi0_base model must be the newest version uploaded to HF hub in LeRobot collection
# POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi05_base/snapshots/9e50c659e8a0a6a3625d111044d1566672399e95"
POLICY_PATH="/gemini/space/users/ysy/project/TEXAS/third_party/RhodesLeRobot/outputs/train_pi05_aw_goodquan_beta0.5_0102/demo_clean_50tasks/26-01-03_15-45-15_pi05_gpu1_ck50_lr5e-5_bs32_s200K_seed42/checkpoints/060000/pretrained_model"

### accelerate launch arguments
GPUS=1 # 4
MAIN_PROCESS_PORT=29510
MIXED_PRECISION=no
GRADIENT_ACCUMULATION_STEPS=1

### Lerobot Training Parameters
BATCH_SIZE=32
TOTAL_STEPS=200000
SAVE_FREQ=20000
# LEARNING_RATE=0.000025
LEARNING_RATE=5e-5
ACTION_CHUNK_SIZE=50
NUM_WORKERS=10
SEED=42
ENFORCE_REPLACE=true
GRADIENT_CHECKPOINTING=true

# ------------------------------------------------------------
### generate output directory
DATE1=$(date "+%y-%m-%d")
TIME1=$(date "+%H-%M-%S")

MODEL_NAME="pi05"
DATASET_NAME="${DATASET_REPO_ID##*/}"
GPU_NUM="gpu${GPUS}"
BS="bs${BATCH_SIZE}"
STEPS_K=$(($TOTAL_STEPS / 1000))
STEPS="s${STEPS_K}K"
LR_SCI=$(printf "%.1e" $LEARNING_RATE | sed 's/e-0/e-/' | sed 's/\.0e/e/')
# For file path, replace decimal point with underscore
LR_PATH=$(echo $LR_SCI | sed 's/\./_/')
LR="lr${LR_PATH}"
CHUNK_SIZE="ck${ACTION_CHUNK_SIZE}"
SEED_STR="seed${SEED}"

OUTPUT_DIR="outputs/train_pi05_aw_goodquan_beta0.5_0102_from60k/${DATASET_NAME}/${DATE1}_${TIME1}_${MODEL_NAME}_${GPU_NUM}_${CHUNK_SIZE}_${LR}_${BS}_${STEPS}_${SEED_STR}"
echo "Output dir: $OUTPUT_DIR"
# ------------------------------------------------------------

cd /gemini/space/users/ysy/project/TEXAS/third_party/RhodesLeRobot
### accelerate launch command
python   src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.revision=$DATASET_REVISION \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_transforms.enable=false \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --enforce_input_output_replace=$ENFORCE_REPLACE \
    --seed=$SEED \
    --policy.path=$POLICY_PATH \
    --policy.push_to_hub=false \
    --policy.chunk_size=$ACTION_CHUNK_SIZE \
    --policy.n_action_steps=$ACTION_CHUNK_SIZE \
    --policy.optimizer_lr=$LEARNING_RATE \
    --policy.gradient_checkpointing=$GRADIENT_CHECKPOINTING \
