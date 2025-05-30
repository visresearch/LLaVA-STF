#!/bin/bash
#SBATCH -o out/mmbench/%j.out ##作业的输出信息文件  
#SBATCH -J mmbench ##作业名  
#SBATCH -w gpu14
#SBATCH -p A6000-ni
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=8

source ~/.bashrc
conda activate llava

SPLIT="mmbench_dev_20230712"
CKPT="llava-v1.5-13b-ours-step6-hid20"

python -m llava.eval.model_vqa_mmbench \
    --model-path /ni_data/users/shencc/tanghao/llava_th/13b/merge2x2_step6_hid20_lr1e-3_e1_4096/ft/llava-vicuna-13b-v1.5 \
    --question-file /ni_data/users/shencc/tanghao/llava_data/eval/mmbench/$SPLIT.tsv \
    --answers-file /ni_data/users/shencc/tanghao/llava_data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /ni_data/users/shencc/tanghao/llava_data/eval/mmbench/$SPLIT.tsv \
    --result-dir /ni_data/users/shencc/tanghao/llava_data/eval/mmbench/answers/$SPLIT \
    --upload-dir /ni_data/users/shencc/tanghao/llava_data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
