#!/bin/sh

python ./mlm.py \
             --model_name_or_path roberta-base \
             --output_dir ../../bertweet-v1.2/ \
             --seed 42 \
             --do_train \
             --train_file ../../bertweet_data/train_data.txt \
             --max_seq_length 512 \
             --num_train_epochs 20 \
             --per_device_train_batch_size 4 \
             --per_device_eval_batch_size 4 \
             --gradient_accumulation_steps 64 \
             --save_steps 100 \
             --cache_dir ../../.cache \
             --fp16 \
             --learning_rate 7e-4 \
             --weight_decay 0.01 \
             --adam_beta1 0.9 \
             --adam_beta2 0.98 \
             --adam_epsilon 1e-6 \
             --lr_scheduler_type linear \
             --warmup_steps 2000 \
             --logging_steps 100 \
             --dataloader_num_workers 4
