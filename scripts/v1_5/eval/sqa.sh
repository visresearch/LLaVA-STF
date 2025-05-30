#!/bin/bash
#SBATCH -o out/sqa/%j.out ##作业的输出信息文件  
#SBATCH -J sqa ##作业名  
#SBATCH -w gpu14
#SBATCH -p A6000-ni
#SBATCH --nodes=1 ##申请1个节点  
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *
#SBATCH --ntasks=8

source ~/.bashrc
conda activate llava

CKPT="llava-v1.5-13b-ours-step6-hid20"

python -m llava.eval.model_vqa_science \
    --model-path /ni_data/users/shencc/tanghao/llava_th/13b/merge2x2_step6_hid20_lr1e-3_e1_4096/ft/llava-vicuna-13b-v1.5 \
    --question-file /ni_data/users/shencc/tanghao/llava_data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /ni_data/users/shencc/tanghao/llava_data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /ni_data/users/shencc/tanghao/llava_data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$CKPT_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$CKPT_result.json
