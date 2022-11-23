#!/usr/bin/env bash -l

# DREAMBOOTH

while getopts u:m:c: flag
do
    case "${flag}" in
        u) uid=${OPTARG};;
        m) model_key=${OPTARG};;
        c) class_key=${OPTARG};;
    esac
done

conda activate diffusers

export UID=$uid # userid
export MODEL_KEY=$model_key # modelxyz
export CLASS_KEY=$class_key # man
export TRAIN_STEPS=800

export MODEL_NAME="$HOME/gpu-instance-s3fs/default-models/stable-diffusion-v1-5"
export MODEL_VAE="$HOME/gpu-instance-s3fs/default-models/models/sd-vae-ft-mse"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export MODEL_VAE="stabilityai/sd-vae-ft-mse"
export OUTPUT_DIR="$HOME/gpu-instance-s3fs/models/$UID"

echo "=================="
echo "RUNNING DREAMBOOTH TRAINING"
echo "=================="

echo "=================="
echo "USER: $UID"
echo "MODEL KEY: $MODEL_KEY"
echo "CLASS KEY: $CLASS_KEY"
echo "=================="

echo "=================="
echo "Convert HEIC to JPG"
echo "=================="
python heictojpg.py "$HOME/gpu-instance-s3fs/uploads/$UID"

echo "=================="
echo "Resize images to 512x512"
echo "=================="
python resize_images.py "$HOME/gpu-instance-s3fs/uploads/$UID"

echo "=================="
echo "Running Training"
echo "=================="
accelerate launch --num_cpu_threads_per_process 8 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path=$MODEL_VAE \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=3434554 \
  --resolution=420 \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --train_text_encoder \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --sample_batch_size=0 \
  --max_train_steps=$TRAIN_STEPS \
  --save_interval=400 \
  --save_sample_prompt="" \
  --n_save_sample=0 \
  --instance_prompt="photo of ${UID}" \
  --class_prompt="photo of a $CLASS_KEY" \
  --instance_data_dir="$HOME/gpu-instance-s3fs/uploads/$UID"\
  --class_data_dir="$HOME/gpu-instance-s3fs/classes/$CLASS_KEY"

#--num_samples=0 \
#--concepts_list="concepts_list.json"
# train_text_encoder Doesn't work with DeepSpeed?

echo "=================="
echo "Convert to SD CPKT"
echo "=================="

# export MODEL_KEY="matt" && export OUTPUT_DIR="../../../../dreambooth/models/$MODEL_KEY" && export TRAIN_STEPS=800 &&
python ../../scripts/convert_diffusers_to_original_stable_diffusion.py --model_path "$OUTPUT_DIR/$TRAIN_STEPS" --checkpoint_path "$OUTPUT_DIR/${MODEL_KEY}.ckpt"