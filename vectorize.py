

import os
import sys
from array import array
import numpy as np

import pathes
import utils


DICT_NUM = 21
features = [None] * DICT_NUM


def load_features():
    max_id = -1
    for i in range(DICT_NUM):
        features[i] = {}
        fn = pathes.path_data + "data%d.txt" % (i+4)
        with open(fn, 'r') as fin:
            ID = 0
            for line in fin:
                line = line.strip()
                features[i][line] = ID
                ID += 1
                if max_id < ID:
                    max_id = ID
        print "# loading features IDs from", fn

    print "# max ID ", max_id



def print_features():
    for b in range(DICT_NUM):
        cnt = 0
        for f in features[b]:
            print "#", b, " - ", f, " -> ", features[b][f]
            cnt += 1
            if cnt > 30:
                print "# break... total", len(features[b])
                break




def write(fout, vec):
    a = array('l', [int(i) for i in vec])
    a.tofile(fout)
    fout.flush()


def vectorize(name):
    # Feature vector:
    # [ ID, click, year, month, day, hour, <24 categorical features> ]
    #
    # For categorical features I will only have 23 indices

    with open(pathes.path_data + name, 'r') as fin:
        # skip header
        fin.readline()
        with open(pathes.path_train + name + ".b", 'wb+') as fout:
            cnt = 0

            for line in fin:
                line = line.strip()
                tokens = line.split(',')
                idx = 0
                vec = []

                vec.append(str(cnt))         # vec id
                idx += 1
                if 24 == len(tokens):
                    vec.append(tokens[idx])  # click
                    idx += 1
                else:
                    vec.append('0')          # this is a test set

                year = tokens[idx][:2]
                month = tokens[idx][2:4]
                day = tokens[idx][4:6]
                hour = tokens[idx][6:]
                vec.append(year)            #
                vec.append(month)           #
                vec.append(day)             #
                vec.append(hour)            #
                idx += 1

                F_IDX = 6

                ff = []
                for i in range(DICT_NUM):
                    k = tokens[idx]
                    if k in features[i]:
                        v = features[i][k]
                    else:
                        v = -1
                    ff.append(v)
                    idx += 1

                cnt += 1

                ##ff.sort()
                vec[F_IDX:F_IDX+DICT_NUM] = ff

                # padding - so I have 32 int items in an array
                vec.extend([0,0,0,0,0])

                write(fout, vec)
            # end for line
            print "# done,", cnt, "lines were written"

def main():
    load_features()
    print_features()
    vectorize('train.csv')
    #vectorize('test.csv')


if __name__ == '__main__':
    main()
