import os
import sys

TASKS = [
    'emoji',
    'emotion',
    'hate',
    'irony',
    'offensive',
    'sentiment',
    'stance']

STANCE_TASKS = [
    'abortion',
    'atheism',
    'climate',
    'feminist',
    'hillary']


def train(args):
    cmd = '''python ./class.py \
        --model_name_or_path {args[model_name_or_path]} \
        --cache_dir ../../.cache \
        --do_train \
        --save_strategy epoch \
        --do_eval \
        --use_fast_tokenizer \
        --num_train_epochs 30 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --train_file {args[train_file]} \
        --validation_file {args[validation_file]} \
        --output_dir {args[output_dir]}'''.format(args=args)
    os.system(cmd)


def main():
    run_args = {'model_name_or_path': sys.argv[1]}
    for task in TASKS:
        if task == 'stance':
            for st in STANCE_TASKS:
                run_args['train_file'] = os.path.join('../ds/tweeteval/datasets/', task, st, 'train.csv')
                run_args['validation_file'] = os.path.join('../ds/tweeteval/datasets/', task, st, 'val.csv')
                run_args['output_dir'] = os.path.join('../../', run_args['model_name_or_path'], task, st + '.txt')
                train(args=run_args)
        else:
            run_args['train_file'] = os.path.join('../ds/tweeteval/datasets/', task, 'train.csv')
            run_args['validation_file'] = os.path.join('../ds/tweeteval/datasets/', task, 'val.csv')
            run_args['output_dir'] = os.path.join('../../', run_args['model_name_or_path'], task + '.txt')
            train(args=run_args)
