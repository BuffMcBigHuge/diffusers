#!/bin/bash -l

# DREAMBOOTH
# git update-index --chmod=+x examples/dreambooth/launch.sh

while getopts u:m:c: flag
do
    case "${flag}" in
        u) uid=${OPTARG};;
        m) model_id=${OPTARG};;
        c) class_key=${OPTARG};;
    esac
done

export USERID=$uid # userid
export MODELID=$model_id # modelid
export CLASS_KEY=$class_key # man
export TRAIN_STEPS=800

export MODEL_NAME="$HOME/default-models/stable-diffusion-v1-5"
export MODEL_VAE="$HOME/default-models/sd-vae-ft-mse"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export MODEL_VAE="stabilityai/sd-vae-ft-mse"

export UPLOADS_OUTPUT_DIR="$HOME/gpu-instance-s3fs/uploads/$USERID"
export MODEL_OUTPUT_DIR="$HOME/gpu-instance-s3fs/models/$USERID/$MODELID"

echo "=================="
echo "RUNNING DREAMBOOTH TRAINING"
python -V
echo "=================="

echo "=================="
echo "USER: $USERID"
echo "MODEL ID: $MODELID"
echo "CLASS KEY: $CLASS_KEY"
echo "=================="

echo "=================="
echo "Convert HEIC to JPG"
echo "=================="
python heictojpg.py "${UPLOADS_OUTPUT_DIR}"

echo "=================="
echo "Resize images to 512x512"
echo "=================="
python resize_images.py "${UPLOADS_OUTPUT_DIR}"

echo "=================="
echo "Running Training"
echo "=================="
accelerate launch --num_cpu_threads_per_process 8 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path=$MODEL_VAE \
  --output_dir=$MODEL_OUTPUT_DIR \
  --revision="fp16" \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=3434554 \
  --resolution=512 \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --train_text_encoder \
  --use_8bit_adam \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --sample_batch_size=0 \
  --max_train_steps=$TRAIN_STEPS \
  --save_interval=$TRAIN_STEPS \
  --n_save_sample=0 \
  --instance_prompt="photo of ${USERID}" \
  --class_prompt="photo of a $CLASS_KEY" \
  --instance_data_dir="$HOME/gpu-instance-s3fs/uploads/$USERID"\
  --class_data_dir="$HOME/gpu-instance-s3fs/classes/$CLASS_KEY"

echo "FINISHED"

# --save_sample_prompt="photo of ${USERID}" \
# --num_samples=0 \
# --concepts_list="concepts_list.json"
# train_text_encoder Doesn't work with DeepSpeed?

# echo "=================="
# echo "Convert to SD CPKT"
# echo "=================="

# convert to ckpt
# python ../../scripts/convert_diffusers_to_original_stable_diffusion.py --model_path "$TRAINING_OUTPUT_DIR/$TRAIN_STEPS" --checkpoint_path "$MODEL_OUTPUT_DIR/${MODEL_KEY}.ckpt"

# get hash
# python gethash.py --checkpoint_path "$MODEL_OUTPUT_DIR/${MODEL_KEY}.ckpt"

# delete training files
# rm -rf $TRAINING_OUTPUT_DIR