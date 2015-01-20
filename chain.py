
import os
import sys
from array import array
import numpy as np
import ctypes
import time
import datetime as dt
import pathes



RF_DLL = ctypes.cdll.LoadLibrary(pathes.path_base + "scripts/chain1/rf.dll")



INDEX = -1
LAST = False
TREES = 1

COLUMNS = 88
K = 16
LNUM = 51

VEC_LEN = 90

TRAIN_DATA = None
TRAIN_BUFFER = 0
TRAIN_BUFFER_SIZE = 0


def log(msg):
    print "#", dt.datetime.now(), "[%d]:" % INDEX, msg
    sys.stdout.flush()

def cmd(msg):
    print msg
    sys.stdout.flush()



def is_master():
    return INDEX == 0


def get_files():
    ends_with = "%d.tmp" % INDEX
    files = [pathes.path_tmp + f for f in os.listdir(pathes.path_tmp) if "tree" == f[:4] and f.endswith(ends_with)]
    return files


def start_process():
    # here I only need just one RF
    rf = RF_DLL.alloc_tree_learner(COLUMNS, K, LNUM)

    for t in range(TREES):
        fname = pathes.path_tmp + "tree_%d_%d.tmp" % (t, INDEX)
        RF_DLL.tofile_tree(ctypes.c_void_p(rf), ctypes.c_char_p(fname))

    RF_DLL.free_tree(ctypes.c_void_p(rf))
# end of start_process


def load_rf(rf, fname):
    RF_DLL.fromfile_tree(ctypes.c_void_p(rf), ctypes.c_char_p(fname))


def need_to_process(rf):
    r = RF_DLL.end_of_splitting(ctypes.c_void_p(rf))
    return 1 != r


def feed_data_to_rf(rf):
    start = dt.datetime.now()

    vectors_num = TRAIN_BUFFER_SIZE / VEC_LEN
    for i in range(vectors_num):
        offset = i * VEC_LEN
        X = ctypes.c_void_p(TRAIN_BUFFER + offset + 2)
        Y = ctypes.c_int(TRAIN_DATA[offset + 1])
        drop_x = ctypes.c_int(0)
        RF_DLL.fit_tree(ctypes.c_void_p(rf), X, Y, ctypes.addressof(drop_x))

    log("# fed %d vectors, time %s sec" % (vectors_num, dt.datetime.now() - start))


def pass_rf_to_next(rf, cur_fname):
    name_part = os.path.basename(cur_fname)
    parts = name_part.split("_")
    new_fname = pathes.path_tmp + "tree_%d_%d.tmp.ren" % (int(parts[1]), INDEX+1)

    RF_DLL.tofile_tree(ctypes.c_void_p(rf), ctypes.c_char_p(new_fname))

    # and finally remane the file to make it available for the next node in the chain
    fname = pathes.path_tmp + "tree_%d_%d.tmp" % (int(parts[1]), INDEX+1)
    os.rename(new_fname, fname)
# end of pass_rf_to_next

def forward_to_master(cur_fname):
    name_part = os.path.basename(cur_fname)
    parts = name_part.split("_")
    new_fname = pathes.path_tmp + "tree_%d_%d.tmp" % (int(parts[1]), INDEX+1)
    fname = pathes.path_tmp + "tree_%d_%d.tmp" % (int(parts[1]), 0)
    os.rename(new_fname, fname)


def stop_process(rf):
    pass

def process():
    # RF to load and teach
    rf = RF_DLL.alloc_tree_learner(COLUMNS, K, LNUM)

    # start process if it's the first node in the chain
    if is_master():
        start_process()

    while True:
        # get files to process
        files = get_files()

        if 0 == len(files):
            log("# no files sleep")
            time.sleep(10)

        else:
            for fname in files:
                load_rf(rf, fname)
                log("# loaded %s" % fname)
                if need_to_process(rf):
                    log("#   processing...")
                    feed_data_to_rf(rf)
                    pass_rf_to_next(rf, fname)
                    log("# removing file")
                    os.remove(fname)
                    if LAST:
                        forward_to_master(fname)
                    else:
                        # tell the next node to start processing
                        cmd("go")

                else:
                    log("# stop processing")
                    stop_process(rf)
        if not is_master():
            break

    # end of while

# end of process



def load_data():
    fname = pathes.path_train + "train.hist.%d" % INDEX

    log("# load data: %s" % fname)

    with open(fname, "rb") as fin:
        fin.seek(0, 2)
        size = fin.tell() / 4   # bytes in file / sizeof int
        fin.seek(0)

        a = array('l')
        a.fromfile(fin, size)

        return a


def params():
    log("# parameters:")

    # TODO
    #if 1 == len(sys.argv):
    #    sys.argv.append("-I1")

    for p in sys.argv[1:]:
        if "-I" == p[:2]:
            global INDEX
            INDEX = int(p[2:])
            log("#   index %d" % INDEX)

        elif "-L" == p or "-LAST" == p.upper():
            global LAST
            LAST = True
            log("#   last in chain %s" % LAST)

        else:
            raise Exception("Unknows parameter: %s" % p)



def main():
    params()

    # just let the first ones to go
    time.sleep(INDEX)

    # train data
    global TRAIN_DATA, TRAIN_BUFFER, TRAIN_BUFFER_SIZE
    TRAIN_DATA = load_data()
    TRAIN_BUFFER, TRAIN_BUFFER_SIZE = TRAIN_DATA.buffer_info()
    log("# read to buffer %d ints" % TRAIN_BUFFER_SIZE)

    if is_master():
        process()
    else:
        # processing loop
        line = sys.stdin.readline()
        while line:
            line = line.strip()

            if "#" == line[0]:
                print line
                sys.stdout.flush()
            elif "STOP" == line:
                break
            elif "get ready" == line:
                # just try to prefetch it from swap
                b = TRAIN_BUFFER[30000]
            elif "go" == line:
                # process chunks
                process()

            line = sys.stdin.readline()
        # end of processing loop

    log("# DONE %d" % INDEX)


if __name__ == '__main__':
    main()
