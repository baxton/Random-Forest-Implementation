
import os
import time
from array import array
import pathes



# id, click, y, m, d, h, [21], 0 0 0 0 0
VEC_LEN_OLD = 32
# id, click, [h + 21][h + 21][h + 21][h + 21]
VEC_LEN_NEW = 90

FEATURES_OLD = 25
FEATURES_NEW = 22

NULL = -999999999


CNT_IDX = 0
CLK_IDX = 1
RATE_IDX = 2

TOTAL_VECTORS = 40428967
VECTORS_PER_FILE = 2000000

def process(fname, hists):
    with open(fname, "rb") as fin:

        cnt_out = 0
        fname_out = pathes.path_train + "train.hist.%d" % cnt_out
        vectors_written = 0
        fout = open(fname_out, "wb+")

        cnt = 0

        try:
            ID = 0
            while True:
                a = array('l')
                a.fromfile(fin, VEC_LEN_OLD)

                o = array('l', [NULL] * VEC_LEN_NEW)

                o[0] = ID
                o[1] = a[1]

                ID += 1

                for f in range(3, FEATURES_OLD):    # 22 features out of 25
                    idx = f + 2
                    fnew = 2 + (f - 3)

                    val = a[idx]

                    o[fnew] = val
                    o[fnew + FEATURES_NEW]   = hists[f][val][CNT_IDX]
                    o[fnew + FEATURES_NEW*2] = hists[f][val][CLK_IDX]
                    o[fnew + FEATURES_NEW*3] = hists[f][val][RATE_IDX]

                o.tofile(fout)
                vectors_written += 1

                if vectors_written == VECTORS_PER_FILE:
                    fout.close()
                    cnt_out += 1
                    fname_out = pathes.path_train + "train.hist.%d" % cnt_out
                    vectors_written = 0
                    fout = open(fname_out, "wb+")

                cnt += 1
                if 0 == (cnt % 100000):
                    time.sleep(2)


        except Exception as ex:
            fout.close()
            print "#", ex, "EOF"




def load_hists():
    hists = [None] * FEATURES_OLD

    cnt = 0

    fname = "hists.csv"
    with open(fname, "r") as fin:
        for line in fin:
            line = line.strip()
            vals = line.split(',')  # <feature id>,<feature val>,<count>,<clicks>,<click / count rate>

            f = int(vals[0])
            v = int(vals[1])
            cnt = int(vals[2])
            clk = int(vals[3])
            rate = int(vals[4])

            if None == hists[f]:
                hists[f] = dict()

            hists[f][v] = (cnt, clk, rate)

            cnt += 1
            if 0 == (cnt % 10000):
                time.sleep(2)

    return hists




def main():
    fname = pathes.path_data + 'sub2\\train.csv.b'
    hists = load_hists()
    process(fname, hists)




main()
