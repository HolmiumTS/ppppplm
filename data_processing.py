# This is used for processing data

import json
import os

data_path = '/run/media/holmium/DATA/TwitterPLM'


def get_path(relate_path):
    return os.path.join(data_path, relate_path)


raw_data = open(file=get_path('2019-06.json'), mode='r')
line_by_text_data = open(file=get_path('2019-06-line.txt'), mode='w')

line = raw_data.readline()

while line:
    data = json.loads(line[:-2])
    line_by_text_data.write(data['text'].replace('\n', ' ') + '\n')
    line = raw_data.readline()
