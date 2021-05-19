# This is used for processing data

import json
import csv

source = open("../ds/semeval18-3a/SemEval2018-T3-train-taskA_emoji.txt", "r")
# train = open("train.csv", 'w')
# val = open("val.csv", 'w')
# traverse through lines one by one

sk = False
data = []
for line in source:
    if not sk:
        sk = True
        continue
    sp = line[:-1].split('\t')
    data.append({'label': sp[1], 'text': sp[2]})

with open('../train.csv', "w", encoding='utf8', newline='') as outFileCsv:
    fileheader = ['label', 'text']
    outDictWriter = csv.DictWriter(outFileCsv, fileheader)
    outDictWriter.writeheader()

    outDictWriter.writerows(data)
    outFileCsv.close()

with open('../val.csv', "w", encoding='utf8', newline='') as outFileCsv:
    fileheader = ['label', 'text']
    outDictWriter = csv.DictWriter(outFileCsv, fileheader)
    outDictWriter.writeheader()

    outDictWriter.writerows(data[-300:])
    outFileCsv.close()
