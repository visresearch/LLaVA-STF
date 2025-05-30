#!/bin/bash

LLAVA_ROOT=root_dir
model_path=./checkpoints/llava-v1.5-vicuna-7b

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

CKPT_NAME=$(basename "llava-ours-ft")
echo "Model path is set to: $model_path"

data_dir=./data/eval_video/MVBench
exp_dir=$LLAVA_ROOT/playground/data/eval
output_dir="${exp_dir}/mvbench_mc/${CKPT_NAME}"

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.video.mvbench_mc \
      --model_path "$model_path" \
      --data_dir "$data_dir" \
      --output_dir "$output_dir" \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX \
      --conv_mode vicuna_v1 &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done