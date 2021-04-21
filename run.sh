./run_mlm.py \
             --model_type roberta \
             --output_dir ./tmp/ \
             --train_file ./data/2019-06-line.txt \
             --per_device_train_batch_size 4 \
             --seed 42 \
             --do_train \
             --tokenizer_name ./tokenizer/ \
             --max_seq_length 512 \
             --max_train_samples 100
