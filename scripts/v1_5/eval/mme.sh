#!/bin/bash
#SBATCH -o out/mme/%j.out ##作业的输出信息文件  
#SBATCH -J mme ##作业名  
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
    --question-file /ni_data/users/shencc/tanghao/llava_data/eval/mme/llava_mme.jsonl \
    --image-folder /ni_data/users/shencc/tanghao/llava_data/eval/mme/MME_Benchmark_release_version \
    --answers-file /ni_data/users/shencc/tanghao/llava_data/eval/mme/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /ni_data/users/shencc/tanghao/llava_data/eval/mme

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
