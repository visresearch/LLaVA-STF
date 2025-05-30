#!/bin/bash
#SBATCH -o out/gqa/%j.out ##作业的输出信息文件  
#SBATCH -J gqa ##作业名  
# SBATCH -w gpu14
#SBATCH -p A6000-ni
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:8 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=64

source ~/.bashrc
conda activate llava

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b-ours-step6-hid20"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/ni_data/users/shencc/tanghao/llava_data/eval/gqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /ni_data/users/shencc/tanghao/llava_th/13b/merge2x2_step6_hid20_lr1e-3_e1_4096/ft/llava-vicuna-13b-v1.5 \
        --question-file /ni_data/users/shencc/tanghao/llava_data/eval/gqa/$SPLIT.jsonl \
        --image-folder /ni_data/users/shencc/tanghao/llava_data/eval/gqa/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
