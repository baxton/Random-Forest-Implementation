

import os
from array import array
import pathes


VEC_LEN = 8
VECTORS_PER_FILE = 3000000


fname = pathes.path_train + "train.csv.b"

fin = open(fname, "rb")
files = [
    open(pathes.path_lr_data + "ent_90_6_51_0(1).b_train.txt", "r"),
    open(pathes.path_lr_data + "ent_90_6_51_0.b_train.txt", "r"),
    open(pathes.path_lr_data + "ent_90_6_51_1(1).b_train.txt", "r"),
    open(pathes.path_lr_data + "ent_90_6_51_1.b_train.txt", "r"),
    open(pathes.path_lr_data + "ent_90_6_51_2(1).b_train.txt", "r"),
    open(pathes.path_lr_data + "ent_90_6_51_2.b_train.txt", "r"),
    open(pathes.path_lr_data + "log_90_8_51_2(11).b_train.txt", "r"),
]

fname_out_pref = pathes.path_lr_data + "lr_train.b"

def main():
    vectors_total = 0

    cnt = 0
    fname_out = "%s.%d" % (fname_out_pref, cnt)
    fout = open(fname_out, "wb+")

    vec = array('f', [0.] * VEC_LEN)

    vectors_written = 0

    try:
        while True:
            a = array('l')
            a.fromfile(fin, 32)

            idx = 0
            vec[idx] = a[1]   # click
            idx += 1

            for f in files:
                line = f.readline().strip()
                while line.startswith("#"):
                    line = f.readline().strip()

                val = float(line)
                vec[idx] = val
                idx += 1

            vec.tofile(fout)
            vectors_total += 1
            vectors_written += 1

            if vectors_written >= VECTORS_PER_FILE:
                vectors_written = 0
                fout.close()
                cnt += 1
                fname_out = "%s.%d" % (fname_out_pref, cnt)
                fout = open(fname_out, "wb+")


    except Exception as ex:
        print ex, "EOF, processed", vectors_total, "vectors"

    fout.close()



if __name__ == '__main__':
    main()
