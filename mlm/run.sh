#!/bin/sh

python ./mlm.py \
             --model_name_or_path roberta-base \
             --output_dir ../../bertweet-v1.1/ \
             --seed 42 \
             --do_train \
             --train_file ../../bertweet_data/train_data.txt \
             --max_seq_length 512 \
             --num_train_epochs 20 \
             --per_device_train_batch_size 4 \
             --per_device_eval_batch_size 4 \
             --gradient_accumulation_steps 63 \
             --save_steps 500 \
             --cache_dir ../../.cache \
             --fp16 \
             --dataloader_num_workers 4
