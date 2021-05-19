#!/bin/bash

python ./run_class_test.py \
        --model_name_or_path ../s18-3a-n/checkpoint-5000/ \
        --cache_dir ../cache \
        --do_train \
        --save_steps 4500 \
        --do_eval \
        --use_fast_tokenizer \
        --num_train_epochs 10 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --train_file ./train.csv \
        --validation_file ./test.csv \
        --output_dir ../test-s18-3a/ \
        --overwrite_output_dir
