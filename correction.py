
import os
import ctypes
from array import array
import numpy as np
from sklearn.ensemble import RandomForestRegressor

RF_DLL = ctypes.cdll.LoadLibrary("./rf.dll")


N = 2000

VEC_LEN = 32



rf_fnames = [
"C:\\Temp\\kaggle\\CTR\\rf\\sub2\\log1_o\\f_90_8_51_0(1).b",
"C:\\Temp\\kaggle\\CTR\\rf\\sub2\\log1_o\\f_90_8_51_0(2).b",
"C:\\Temp\\kaggle\\CTR\\rf\\sub2\\log1_o\\f_90_8_51_0(3).b",
"C:\\Temp\\kaggle\\CTR\\rf\\sub2\\log1_o\\f_90_8_51_0(4).b",
"C:\\Temp\\kaggle\\CTR\\rf\\sub2\\log1_o\\f_90_8_51_0(5).b",
"C:\\Temp\\kaggle\\CTR\\rf\\sub2\\log1_o\\f_90_8_51_0(6).b",
#"C:\\Temp\\kaggle\\CTR\\rf\\sub2\\log1_o\\f_90_8_51_0(7).b",
#"C:\\Temp\\kaggle\\CTR\\rf\\sub2\\log1_o\\f_90_8_51_0(8).b",
#"C:\\Temp\\kaggle\\CTR\\rf\\sub2\\log1_o\\f_90_8_51_0(9).b",
]
trees = []

data_file_names = [
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.0",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.1",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.2",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.3",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.4",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.5",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.6",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.7",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.8",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.9",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.10",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.11",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.12",
"C:\\Temp\\kaggle\\CTR\\data\\sub2\\train.csv.b.13",
]
data_files = []

VECTORS_NUM = N * len(data_file_names)
BUFFER_LEN = VEC_LEN * VECTORS_NUM
X = np.array([0]*BUFFER_LEN, dtype=np.int32)



def get_X():
    global X

    #
    for fn in data_file_names:
        data_files.append(open(fn, "rb"))

    #
    idx = 0
    for fin in data_files:
        a = array('l')
        a.fromfile(fin, VEC_LEN * N)
        X[idx:idx+VEC_LEN * N] = a[:]
        idx += VEC_LEN * N

    #
    X = X.reshape((VECTORS_NUM, VEC_LEN))



def predict(x):
    sum = 0.
    p = ctypes.c_double(0.)
    for t in range(len(trees)):
        RF_DLL.predict(ctypes.c_void_p(trees[t]), x[2:].ctypes.data, ctypes.addressof(p))
        sum += p.value

    return sum / len(trees)


def process():
    global X

    name = "result.txt"

    #
    t = 0
    for fn in rf_fnames:
        trees.append(RF_DLL.alloc_tree())
        RF_DLL.fromfile_tree(ctypes.c_void_p(trees[t]), ctypes.c_char_p(fn))
        t += 1

    # Y
    Y = np.array([0] * VECTORS_NUM, dtype=float)
    idx = 0
    for x in X:
        p = predict(x[2:])
        Y[idx] = p - x[1]
        idx += 1
    print "# Y is prepared"

    #
    c = RandomForestRegressor(n_estimators=50)
    c.fit(X[:,2:], Y)
    print "# fit is done"

    X = None

    #
    file_size = 585915392
    items = file_size / 4

    items_num_1 = items / 3
    items_num_2 = items_num_1
    items_num_3 = items - items_num_1 - items_num_2

    with open(r'C:\Temp\kaggle\CTR\data\sub2\test.csv.b', 'rb') as fin:
        with open(".\\" + name, "w+") as fout:
            fout.write("# %s%s" % (name, os.linesep))

            a = array('l')
            a.fromfile(fin, items_num_1)
            vectors = items_num_1 / VEC_LEN
            for v in range(vectors):
                idx = int(v * VEC_LEN)
                x = np.array(a[idx:idx+VEC_LEN], dtype=np.int32)

                p = predict(x)
                delta = c.predict(x[2:])
                y = p - delta
                y = y if y >= 0. else 0.
                y = y if y <= 1. else 1.

                fout.write("%2.16f%s" % (y, os.linesep))
                fout.flush()
            a = None
            a = array('l')
            a.fromfile(fin, items_num_2)
            vectors = items_num_2 / VEC_LEN
            for v in range(vectors):
                idx = int(v * VEC_LEN)
                x = np.array(a[idx:idx+VEC_LEN], dtype=np.int32)

                p = predict(x)
                delta = c.predict(x[2:])
                y = p - delta
                y = y if y >= 0. else 0.
                y = y if y <= 1. else 1.

                fout.write("%2.16f%s" % (y, os.linesep))
                fout.flush()
            a = None
            a = array('l')
            a.fromfile(fin, items_num_3)
            vectors = items_num_3 / VEC_LEN
            for v in range(vectors):
                idx = int(v * VEC_LEN)
                x = np.array(a[idx:idx+VEC_LEN], dtype=np.int32)

                p = predict(x)
                delta = c.predict(x[2:])
                y = p - delta
                y = y if y >= 0. else 0.
                y = y if y <= 1. else 1.

                fout.write("%2.16f%s" % (y, os.linesep))
                fout.flush()

    print "# DONE"


def main():
    get_X()
    process()



if __name__ == '__main__':
    main()
