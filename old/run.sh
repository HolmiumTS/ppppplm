#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=plm_tra
#SBATCH --gres=gpu:1
#SBATCH --output=train_model_v1.out
#SBATCH --get-user-env

./run_mlm.py \
             --model_type roberta \
             --output_dir ./v1/ \
             --train_file ./data/2019-06-line-train.txt \
             --per_device_train_batch_size 4 \
             --seed 42 \
             --do_train \
             --tokenizer_name ./tokenizer/ \
             --max_seq_length 512 \
             --evaluation_strategy epoch \
             --num_train_epochs 20 \
             --do_eval \
             --validation_file ./data/2019-06-line-eval.txt \
             --save_steps 50000 \
             --fp16
