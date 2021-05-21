import os
from normal import BertweetTokenizer
import csv

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

ttt = BertweetTokenizer()


def preprocess(labels, text, output):
    lf = open(labels, 'r').readlines()
    tf = open(text, 'r').readlines()
    of = open(output, 'w')
    dt = []
    ll = len(lf)
    for i in range(ll):
        dt.append({'label': lf[i][:-1], 'text': ttt.normalizeTweet(tf[i][:-1])})
    fileheader = ['label', 'text']
    outDictWriter = csv.DictWriter(of, fileheader)
    outDictWriter.writeheader()


def main():
    run_args = {}
    for task in TASKS:
        if task == 'stance':
            for st in STANCE_TASKS:
                for s in ['train', 'val']:
                    labels = os.path.join('../ds/tweeteval/datasets/', task, st, s + '_labels.txt')
                    text = os.path.join('../ds/tweeteval/datasets/', task, st, s + '_text.txt')
                    output = os.path.join('../ds/tweeteval/datasets/', task, st, s + '.txt')
                    preprocess(labels, text, output)
        else:
            for s in ['train', 'val']:
                run_args['labels'] = os.path.join('../ds/tweeteval/datasets/', task, s + '_labels.txt')
                run_args['text'] = os.path.join('../ds/tweeteval/datasets/', task, s + '_text.txt')
                run_args['output'] = os.path.join('../ds/tweeteval/datasets/', task, s + '.txt')
                preprocess(**run_args)
