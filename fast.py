
import os
import sys
import mmap
from array import array
import datetime as dt

import numpy as np
import scipy as sp

import ctypes


import pathes
import utils




VEC_LEN = 32
VEC_LEN_BYTES = VEC_LEN * 4


TOTAL_FEATURES = 4 + 21

TREES   = 1
COLUMNS = TOTAL_FEATURES
K       = TOTAL_FEATURES / 2
LNUM    = 11


FOR_BS = 60

LINES_IN_TRAIN = 40428967
LINES_IN_BS    = int(LINES_IN_TRAIN * (FOR_BS / 100.))
#LINES_IN_BS    = 32343173    # 80%
#LINES_IN_BS    = 40428967    # 100%
#LINES_IN_BS    = 36386070    # 90%
#LINES_IN_BS    = 100


RF_DLL = ctypes.cdll.LoadLibrary(pathes.path_base + "scripts/sub2/rf.dll")


train_fname = pathes.path_train + 'train.csv.b'

tree = RF_DLL.alloc_tree_learner(COLUMNS, K, LNUM)
bootstraps = set(sp.random.randint(0, LINES_IN_TRAIN, LINES_IN_BS))

def process_vectors(arr, length_bytes):
    #data = np.fromstring( arr, dtype=np.int32 )
    #data = data.reshape((data.shape[0]/VEC_LEN, VEC_LEN)).astype(float)

    data_int = array('l', arr[:])
    data = array('d', data_int)

    rows = len(data) / VEC_LEN

    for i in range(rows):
        idx = i * VEC_LEN
        print data[idx]
        ID = data[idx+0]
        if ID in bootstraps:
            drop_x = ctypes.c_int(0)
            vec_addr = data.buffer_info()[0] + i * VEC_LEN_BYTES
            y = data[idx+1]
            RF_DLL.fit_tree(tree, ctypes.c_void_p(vec_addr + 2), ctypes.c_double(y), ctypes.addressof(drop_x));


def process():

    total_start = dt.datetime.now()

    with open(train_fname, "rb") as fin:
        # get file size
        fin.seek(0, 2)
        file_size = fin.tell()
        fin.seek(0)
        print "# file size", file_size

        iter_num = 0
        while True:
            iter_num += 1
            iter_start = dt.datetime.now()

            l = 2 ** 22     # vector length in bytes is 128b so I have 32768 vectors in block
                            # the rest of my train.csv.b will be 3330944 bytes which gives me extra 26023 vectors
            o = 0

            while (o + l) < file_size:
                m=mmap.mmap(fin.fileno(), l, access=1, offset=o)
                # process vectors
                process_vectors(m, l)

                # move to the next block
                o += l

            # process the rest
            l = file_size - o
            m=mmap.mmap(fin.fileno(), l, access=1, offset=o)
            print "# last offset", o, "length", l
            # process vectors
            process_vectors(m, l)

            iter_stop = dt.datetime.now()
            print "# iteration", iter_num, "time", (iter_stop - iter_start)

            RF_DLL.stop_fit_tree(tree)
            if 1 == RF_DLL.end_of_splitting(tree):
                break
        m.close()

    # save tree in a file
    fpref = "%d_%d_%d" % (FOR_BS, TREES, LNUM)
    rf_fname = pathes.path_base + "rf\\sub2\\" + fpref + ".b"
    cnt = 0
    while os.path.exists(rf_fname):
        cnt += 1
        rf_fname = pathes.path_base + "rf\\sub2\\" + fpref + ("(%d)" % cnt) + ".b"
    RF_DLL.tofile_tree(tree, ctypes.c_char_p(rf_fname))

    # done
    print "# DONE total time", (dt.datetime.now() - total_start)






def main():
    process()





if __name__ == '__main__':
    main()






#
# paste -d, rf_results.txt 60_1_5_0.b.result.txt > new_rf_results.txt
#
#








