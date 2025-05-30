#!/bin/bash
#SBATCH -o out/pope/%j.out ##作业的输出信息文件  
#SBATCH -J pope ##作业名  
#SBATCH -w gpu14
#SBATCH -p A6000-ni
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=8

source ~/.bashrc
conda activate llava

CKPT="llava-v1.5-13b-ours-step6-hid20"

python -m llava.eval.model_vqa_loader \
    --model-path /ni_data/users/shencc/tanghao/llava_th/13b/merge2x2_step6_hid20_lr1e-3_e1_4096/ft/llava-vicuna-13b-v1.5 \
    --question-file /ni_data/users/shencc/tanghao/llava_data/eval/pope/llava_pope_test.jsonl \
    --image-folder /public/tanghao/dataset/coco_2014/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /ni_data/users/shencc/tanghao/llava_data/eval/pope/coco \
    --question-file /ni_data/users/shencc/tanghao/llava_data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl
