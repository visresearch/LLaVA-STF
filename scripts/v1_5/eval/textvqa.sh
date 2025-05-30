#!/bin/bash
#SBATCH -o out/textvqa/%j.out ##作业的输出信息文件  
#SBATCH -J textvqa ##作业名  
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
    --question-file /ni_data/users/shencc/tanghao/llava_data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /ni_data/users/shencc/tanghao/llava_data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /ni_data/users/shencc/tanghao/llava_data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl
