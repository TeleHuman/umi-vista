#!/bin/bash

echo current python: $(which python)
echo current accelerate: $(which accelerate)

gpu_id=0
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Dataset configuration
# DATASET_REPO_ID="HuggingFaceVLA/libero"

# DATASET_REPO_ID="RoboTwin2/v30/demo_clean_50tasks"
# DATASET_REPO_ID="IPEC-COMMUNITY/bridge_orig_lerobot"
# DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"
# DATASET_REVISION="v3.0"
# DATASET_REVISION=null

# Pretrained model configuration  
### the pi0_base model must be the newest version uploaded to HF hub in LeRobot collection
POLICY_PATH="$HF_HUB_CACHE/models--lerobot--pi05_base/snapshots/9e50c659e8a0a6a3625d111044d1566672399e95"
# POLICY_PATH="/gemini/platform/public/embodiedAI/huggingface_cache/hub/models--lerobot--pi05_libero_finetuned/snapshots/d8419fc249cbb1f29b0c528f05c0d2fe50f46855"

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
NUM_WORKERS=16
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
# ------------------------------------------------------------

# for policy
advantage_beta=0.1

########### robotwin #################################
bench="robotwin"
DATASET_REPO_ID="RoboTwin2/v30/demo_clean_50tasks"
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO_ID"
DATASET_REVISION="v3.0"
# mc
critic_model_ckpt="/gemini/platform/public/embodiedAI/huggingface_cache/hub/models--breezeyoung--texas_robotwin2_MC_80k/snapshots/33dbdd76ba5a30833064d35ee37c6e22bfad0cc5"
OUTPUT_DIR="outputs/temp/aaa-0121/train_pi05_${bench}_beta${advantage_beta}_good_q_critic_mc/${DATASET_NAME}/${DATE1}_${TIME1}_${MODEL_NAME}_${GPU_NUM}_${CHUNK_SIZE}_${LR}_${BS}_${STEPS}_${SEED_STR}"
# # td
# critic_model_ckpt="/gemini/platform/public/embodiedAI/huggingface_cache/hub/models--breezeyoung--texas_robotwin2_TD_80k/snapshots/a1ae91e5e63eadee0f2a3f9196a12dc0aa299ebf"
# OUTPUT_DIR="outputs/train_pi05_${bench}_beta${advantage_beta}_good_critic_td/${DATASET_NAME}/${DATE1}_${TIME1}_${MODEL_NAME}_${GPU_NUM}_${CHUNK_SIZE}_${LR}_${BS}_${STEPS}_${SEED_STR}"

# OUTPUT_DIR="outputs/temp"

echo "Output dir: $OUTPUT_DIR"

cd /gemini/space/users/ysy/project/TEXAS/third_party/RhodesLeRobot
### accelerate launch command
python   src/lerobot/scripts/lerobot_train_good_opt.py \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --dataset.revision=$DATASET_REVISION \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_transforms.enable=false \
    --policy.path=$POLICY_PATH \
    --policy.push_to_hub=false \
    --policy.chunk_size=$ACTION_CHUNK_SIZE \
    --policy.n_action_steps=$ACTION_CHUNK_SIZE \
    --policy.optimizer_lr=$LEARNING_RATE \
    --policy.gradient_checkpointing=$GRADIENT_CHECKPOINTING \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE --steps=$TOTAL_STEPS --save_freq=$SAVE_FREQ --num_workers=$NUM_WORKERS \
    --enforce_input_output_replace=$ENFORCE_REPLACE \
    --seed=$SEED \
    --policy.critic_model_ckpt=$critic_model_ckpt \
    --policy.advantage_beta=$advantage_beta \
    --policy.bench=$bench \
    
    # --dataset.critic_model_ckpt=$critic_model_ckpt