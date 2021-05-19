#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=bertweet
#SBATCH --gres=gpu:rtx:1
#SBATCH --output=train_bertweet.out
#SBATCH --get-user-env

./run_mlm_bertweet.py \
             --model_type roberta \
             --output_dir ../bertweet-v1/ \
             --train_file ../bertweet_data/train_data.txt \
             --seed 42 \
             --do_train \
             --tokenizer_name vinai/bertweet-base \
             --max_seq_length 512 \
             --num_train_epochs 20 \
             --save_steps 50 \
             --cache_dir ../.cache \
             --fp16 \
             --preprocessing_num_workers 5 \
             --dataloader_num_workers 4
