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


def predict(args):
    cmd = '''python ./class.py \
        --model_name_or_path {args[model_name_or_path]} \
        --cache_dir ../../.cache \
        --do_predict \
        --use_fast_tokenizer \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --test_file {args[test_file]} \
        --task_name {args[task_name]} \
        --output_dir {args[output_dir]}'''.format(args=args)
    os.system(cmd)


def main():
    root = sys.argv[1]
    run_args = {}
    for task in TASKS:
        if task == 'stance':
            for st in STANCE_TASKS:
                run_args['test_file'] = os.path.join('../ds/tweeteval/datasets/', task, st, 'test.csv')
                run_args['model_name_or_path'] = os.path.join('../../', run_args['model_name_or_path'], task, st) + '/'
                run_args['output_dir'] = os.path.join(root, task)
                run_args['task_name'] = st
                predict(args=run_args)
        else:
            run_args['test_file'] = os.path.join('../ds/tweeteval/datasets/', task, 'test.csv')
            run_args['model_name_or_path'] = os.path.join('../../', run_args['model_name_or_path'], task) + '/'
            run_args['output_dir'] = os.path.join(root)
            run_args['task_name'] = task
            predict(args=run_args)


if __name__ == '__main__':
    main()
