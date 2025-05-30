#!/bin/bash  
#SBATCH -o out/vqav2.%j.out ##作业的输出信息文件  
#SBATCH -J vqav2 ##作业名  
# SBATCH -w gpu16
#SBATCH -p A6000-ni
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:8 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=64

# exec 2>out/std_vqav2.out

source ~/.bashrc
conda activate llava

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b-ours-step6-hid20"
SPLIT="llava_vqav2_mscoco_test-dev2015"

ROOT="/ni_data/users/shencc/tanghao/llava_data"
# --model-base /public/tanghao/dataset/llava/vllm/llm/vicuna-7b-v1.5 \
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /ni_data/users/shencc/tanghao/llava_th/13b/merge2x2_step6_hid20_lr1e-3_e1_4096/ft/llava-vicuna-13b-v1.5 \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder $ROOT/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --dir ./playground/data/eval/vqav2 --split $SPLIT --ckpt $CKPT

