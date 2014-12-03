


import os
import sys
from array import array

import numpy as np
import scipy as sp

import ctypes




import pathes
import utils



RF_DLL = ctypes.cdll.LoadLibrary(pathes.path_base + "scripts/sub2/rf.dll")


#train_fname = pathes.path_train + 'train.csv.b'
test_fname = pathes.path_train + 'test.csv.b'

tree = RF_DLL.alloc_tree()


def process(rf_fname):
    name = os.path.basename(rf_fname)
    print name

    test_result = pathes.path_base + name + '_result.txt'
    if os.path.exists(test_result):
        return

    # load classifier
    RF_DLL.fromfile_tree(ctypes.c_void_p(tree), ctypes.c_char_p(rf_fname))

    # read test data and classify
    with open(test_result, 'w+') as fout:
        fout.write("%s%s" % (name, os.linesep))
        with open(test_fname, "rb") as fin:
            try:
                while True:
                    a = array('l')
                    a.fromfile(fin, 32)      # 32 int for 1 vector

                    ID = a[0]
                    X = np.array(a[2:], dtype=np.float64)
                    p = ctypes.c_double(0.)

                    RF_DLL.predict(ctypes.c_void_p(tree), X.ctypes.data, ctypes.addressof(p))

                    #print ID, p.value
                    fout.write("%f%s" % (p.value, os.linesep))
            except:
                pass  # EOF











def main():
    if 2 == len(sys.argv):
        rf_fname = sys.argv[1]
        process(rf_fname)
    else:
        #print "# ERROR random forest file name is not specified"
        #return 0
        files = [pathes.path_base + "rf\\sub2\\" + fn for fn in os.listdir(pathes.path_base + "rf\\sub2\\") if fn[-2:] == ".b"]
        for fn in files:
            process(fn)





if __name__ == '__main__':
    main()






#
# paste -d, rf_results.txt 60_1_5_0.b.result.txt > new_rf_results.txt
#
#








