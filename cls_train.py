


import os
import sys
from array import array

import numpy as np
import scipy as sp

import ctypes




import pathes
import utils



RF_DLL = ctypes.cdll.LoadLibrary(pathes.path_base + "scripts/log1/rf.dll")


fname = pathes.path_train + 'train.csv.b'

tree = RF_DLL.alloc_tree()


def process(rf_fname):
    with open(fname, "rb") as fin:
        fin.readline()  # skip header

        name = os.path.basename(rf_fname)
        print name

        test_result = pathes.path_base + name + '_train.txt'
        if os.path.exists(test_result):
            return

        # load classifier
        RF_DLL.fromfile_tree(ctypes.c_void_p(tree), ctypes.c_char_p(rf_fname))

        # read test data and classify
        with open(test_result, 'w+') as fout:
            fout.write("%s%s" % (name, os.linesep))

            try:
                while True:
                    a = array('l')
                    a.fromfile(fin, 32)
                    X = np.array(a[2:], dtype=np.int32)

                    p = ctypes.c_double(0.)

                    RF_DLL.predict(ctypes.c_void_p(tree), X.ctypes.data, ctypes.addressof(p))

                    fout.write("%f%s" % (p.value, os.linesep))
            except Exception as ex:
                print ex, "EOF"










def main():

    files = [pathes.path_base + "rf\\sub2\\" + fn for fn in os.listdir(pathes.path_base + "rf\\sub2\\") if fn[-2:] == ".b"]

    if 2 == len(sys.argv):
        files = sys.argv[1].split(",")

    for fn in files:
        process(fn)





if __name__ == '__main__':
    main()






#
# paste -d, rf_results.txt 60_1_5_0.b.result.txt > new_rf_results.txt
#
#








