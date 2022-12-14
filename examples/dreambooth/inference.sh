#!/bin/bash -l

# INFERENCE
# git update-index --chmod=+x examples/dreambooth/inference.sh

while getopts u:m:t:p:n:s:q:h:w:c: flag
do
    case "${flag}" in
        u) uid=${OPTARG};;
        m) model_id=${OPTARG};;
        t) theme_id=${OPTARG};;
        p) prompt=${OPTARG};;
        n) negative_prompt=${OPTARG};;
        s) steps=${OPTARG};;
        q) samples=${OPTARG};;
        h) height=${OPTARG};;
        w) width=${OPTARG};;
        c) cfg=${OPTARG};;
    esac
done

export MODEL_DIR="$HOME/gpu-instance-s3fs/models/$uid/${model_id}"
export IMAGE_OUTPUT_DIR="$HOME/gpu-instance-s3fs/outputs/$uid/${model_id}/${theme_id}"

echo "=================="
echo "RUNNING INFERENCE"
python -V
echo "=================="

echo "=================="
echo "USER: $uid"
echo "MODEL ID: $model_id"
echo "THEME ID: $theme_id"
echo "PROMPT": "$prompt"
echo "NEGATIVE PROMPT": "$negative_prompt"
echo "STEPS": "$steps"
echo "SAMPLES": "$samples"
echo "HEIGHT": "$height"
echo "WIDTH": "$width"
echo "CFG": "$cfg"
echo "=================="

python inference.py \
    --prompt "$prompt" \
    --negative_prompt "$negative_prompt" \
    --ddim_steps "$steps" \
    --n_samples "$samples" \
    --height "$height" \
    --width "$width" \
    --cfg "$cfg" \
    --modeldir "$MODEL_DIR" \
    --outdir "$IMAGE_OUTPUT_DIR"

echo "FINISHED"