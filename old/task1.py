import normal
import sys
import json


def main():
    ttt = normal.BertweetTokenizer()
    i = open(sys.argv[1], 'r')
    o = open(sys.argv[2], 'w')
    for line in i:
        o.write(json.dumps({'text': ttt.normalizeTweet(line)}))
        o.write('\n')


if __name__ == '__main__':
    main()
