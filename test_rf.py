


import os
import array
import numpy as np
import scipy as sp

from sklearn.ensemble import RandomForestClassifier


THRESHOLD = .5
VEC_LEN = 10 * 1


def load_data(fname):
    result = []
    with open(fname, "rb") as fin:
        fin.seek(0, 2)
        size = fin.tell() / 4
        print "File size: ", size
        fin.seek(0)
        bytes_read = 0
        try:
            while bytes_read < size:
                a = array.array("d")
                a.fromfile(fin, VEC_LEN)
                result.append(list(a))
                bytes_read += VEC_LEN
                #print "Read: ", bytes_read
        except:
            pass
    return np.array(result, dtype=int)


def test(rf, data):
    TP = 0.
    FP = 0.
    TN = 0.
    FN = 0.
    YSUM = 0.

    for r in range(data.shape[0]):
        y = data[r,0]
        p = rf.predict(data[r, 1:])[0]

        p = 1 if p > THRESHOLD else 0

        if (p == 1.      and y == 1.):
            TP += 1.
        elif (p == 1. and y == 0.):
            FP += 1.
        elif (p == 0. and y == 1.):
            FN += 1.
        elif (p == 0. and y == 0.):
            TN += 1.

        YSUM += y

    print "Total positives: ", (TP + FP), "; Total negatives: ", (TN + FN)

    sensitivity = TP / (TP + FN)
    precision    = TP / (TP + FP)
    F1 = 2. * (sensitivity * precision) / (sensitivity + precision)

    print "True Pos: ", TP, "; False Pos: ", FP
    print "Sens: ", sensitivity, ", prec: ", precision, ", F1: ", F1, ", Total ones: ", YSUM, ", Total: ", data.shape[0]




def main():
    fname_train = "C:\\Temp\\kaggle\\epilepsy\\data\\dog_train.bin"
    fname_test = "C:\\Temp\\kaggle\\epilepsy\\data\\dog_cross_test.bin"

    train_data = load_data(fname_train)
    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(train_data[:,1:], train_data[:,0])

    test_data = load_data(fname_test)
    test(rf, test_data)


if __name__ == '__main__':
    main()
