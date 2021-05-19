#!/bin/bash

python ./run_class.py \
        --model_name_or_path ../checkpoint-68000/ \
        --cache_dir ../cache \
        --do_train \
        --save_steps 5000 \
        --do_eval \
        --use_fast_tokenizer \
        --num_train_epochs 20 \
        --per_device_train_batch_size 6 \
        --per_device_eval_batch_size 6 \
        --train_file ./train.csv \
        --validation_file ./val.csv \
        --output_dir ../s18-3a-n/