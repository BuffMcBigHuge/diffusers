# DREAMBOOTH

export MODEL_KEY="anatolie"
export CLASS_KEY="man"
export TRAIN_STEPS=800
export TOKEN="ertpbrfklztr"

export MODEL_NAME="models/stable-diffusion-v1-5"
export MODEL_VAE="models/sd-vae-ft-mse"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# export MODEL_VAE="stabilityai/sd-vae-ft-mse"
export OUTPUT_DIR="$HOME/gpu-instance-s3fs/models/$MODEL_KEY"

# convert HEIC to JPG
python heictojpg.py "$HOME/gpu-instance-s3fs/uploads/$MODEL_KEY"
# resize images to 512x512
python resize_images.py "$HOME/gpu-instance-s3fs/uploads/$MODEL_KEY"

accelerate launch --num_cpu_threads_per_process 8 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path=$MODEL_VAE \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=3434554 \
  --resolution=400 \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --train_text_encoder \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --sample_batch_size=4 \
  --max_train_steps=$TRAIN_STEPS \
  --save_interval=$TRAIN_STEPS \
  --save_sample_prompt="photo of ${MODEL_KEY}${TOKEN}" \
  --instance_prompt="photo of ${MODEL_KEY}${TOKEN}" \
  --class_prompt="photo of a $CLASS_KEY" \
  --instance_data_dir="$HOME/gpu-instance-s3fs/uploads/$MODEL_KEY"\
  --class_data_dir="$HOME/gpu-instance-s3fs/classes/$CLASS_KEY"

#--concepts_list="concepts_list.json"
# train_text_encoder Doesn't work with DeepSpeed?

# Convert to SD CPKT
# export MODEL_KEY="matt" && export OUTPUT_DIR="../../../../dreambooth/models/$MODEL_KEY" && export TRAIN_STEPS=800 &&
python ../../scripts/convert_diffusers_to_original_stable_diffusion.py --model_path "$OUTPUT_DIR/$TRAIN_STEPS" --checkpoint_path "$OUTPUT_DIR/${MODEL_KEY}_model.ckpt"