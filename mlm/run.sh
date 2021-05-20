#!/bin/sh

python ./mlm.py \
             --model_name_or_path roberta-base \
             --output_dir ../../bertweet-v1/ \
             --seed 42 \
             --do_train \
             --do_eval \
             --train_file ../../bertweet_data/train_data.txt \
             --validation_file ../../bertweet_data/train_data.txt \
             --max_seq_length 512 \
             --num_train_epochs 20 \
             --per_device_train_batch_size 6 \
             --per_device_eval_batch_size 6 \
             --gradient_accumulation_steps 42 \
             --save_steps 5000 \
             --cache_dir ../../.cache \
             --evaluation_strategy epoch \
             --fp16 \
             --dataloader_num_workers 4