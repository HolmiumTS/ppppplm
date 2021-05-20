./run_mlm_bertweet.py \
             --model_type roberta \
             --output_dir ../bertweet-v1/ \
             --train_file ../bertweet_data/train_data.txt \
             --per_device_train_batch_size 4 \
             --seed 42 \
             --do_train \
             --tokenizer_name vinai/bertweet-base \
             --max_seq_length 512 \
             --num_train_epochs 3 \
             --save_steps 5 \
             --cache_dir ../.cache \
             --fp16