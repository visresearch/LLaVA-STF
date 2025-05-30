#!/bin/bash
#SBATCH -o out/vizwiz/%j.out ##作业的输出信息文件  
#SBATCH -J vizwiz ##作业名  
# SBATCH -w gpu16
#SBATCH -p A6000-ni
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:8 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=64

source ~/.bashrc
conda activate llava

CKPT="llava-v1.5-13b-ours-step6-hid20"

python -m llava.eval.model_vqa_loader \
    --model-path /ni_data/users/shencc/tanghao/llava_th/13b/merge2x2_step6_hid20_lr1e-3_e1_4096/ft/llava-vicuna-13b-v1.5 \
    --question-file /ni_data/users/shencc/tanghao/llava_data/eval/vizwiz/llava_test.jsonl \
    --image-folder /ni_data/users/shencc/tanghao/llava_data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /ni_data/users/shencc/tanghao/llava_data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$CKPT.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$CKPT.json
