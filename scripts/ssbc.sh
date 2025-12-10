#!/bin/bash
PYTHON="/home/hpo/anaconda3/envs/data8014/bin/python3"
export CUDA_VISIBLE_DEVICES=${1}

DATASETS=("cubc")
TASKs=("A_L+A_U+B->A_U+B+C")

for d in ${!DATASETS[@]}; do
    for t in ${!TASKs[@]}; do
        echo ${DATASETS[$d]}
        ${PYTHON} -m train \
                    --dataset_name ${DATASETS[$d]} \
                    --batch_size 150 \
                    --grad_from_block 11 \
                    --epochs 200 \
                    --num_workers 8 \
                    --sup_weight 0.35 \
                    --weight_decay 5e-5 \
                    --warmup_teacher_temp 0.07 \
                    --teacher_temp 0.04 \
                    --warmup_teacher_temp_epochs 30 \
                    --memax_weight 2 \
                    --transform 'imagenet' \
                    --lr 0.05 \
                    --eval_funcs 'v2' \
                    --src_env 'N/A' \
                    --tgt_env 'N/A' \
                    --aux_env 'N/A' \
                    --task_type ${TASKs[$t]} \
                    --exp_name 'cubc+hilo'
    done
done
