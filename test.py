

import os
import sys
import array
import numpy as np
import scipy as sp
import scipy.io as sio
import numpy.fft as fft
import ctypes
import datetime as dt

import scipy.spatial.distance as spd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import files
import features

#DLL = ctypes.cdll.LoadLibrary("C:\\Temp\\kaggle\\epilepsy\\scripts\\dtw_fast.dll")
RF_DLL = ctypes.cdll.LoadLibrary("C:\\Temp\\kaggle\\epilepsy\\scripts\\sub1\\rf.dll")


path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_train = path + "prepared_sub3\\"


interictal_prefix = "interictal_segment"
preictal_prefix = "preictal_segment"
test_prefix = "test_segment"


PREICTAL_CLS = 1
INTERICTAL_CLS = 0


X_LEN = features.X_LEN





test_ii_indices = [69,287,459,369,234,457,414,297,235,415,825,703,691,801,820,673,608,624,552,798,1725,1699,1063,2161,2021,2318,2419,2014,2399,1682,3171,2678,3136,2436,3193,2947,2568,2602,2876,2983,3581,3597,3240,3325,3469,3249,3454,3378,3311,3649,3708,3688,3705,3712,3699,3716,3690,3722,3686,3697,3747,3765,3726,3742,3728,3751,3737,3736,3733,3756]
test_pi_indices = [3789,3768,3780,3770,3772,3774,3775,3776,3787,3779,3811,3824,3827,3810,3829,3795,3796,3816,3820,3831,3866,3870,3846,3868,3873,3854,3894,3869,3885,3875,3904,3961,3920,3983,3926,3916,3922,3959,3929,3953,4002,4021,4022,4006,4019,4009,4011,4027,4015,4018,4031,4033,4035,4038,4041,4044,4046,4047,4049,4051,4055,4056,4058,4060,4062,4065]




class RF(object):
    def __init__(self, trees, k, lnum):
        self.trees = trees
        self.k = k
        self.lnum = lnum



    def alloc(self):
        RF_DLL.alloc_rf(ctypes.c_int(self.trees), ctypes.c_int(self.k), ctypes.c_int(self.lnum))

    def dealloc(self):
        RF_DLL.dealloc_rf()

    def fit(self, X, Y):
        rows = X.shape[0]
        columns = X.shape[1]
        RF_DLL.fit_rf(X.ctypes.data, Y.ctypes.data, ctypes.c_int(rows), ctypes.c_int(columns))

    def predict(self, X):
        X = np.array(X, dtype=float)
        rows = X.shape[0]
        columns = X.shape[1]
        res = []
        for r in range(rows):
            v = X[r, :]
            p = ctypes.c_double(0.)
            RF_DLL.predict_rf(v.ctypes.data, ctypes.c_int(columns), ctypes.addressof(p))

            p = p.value
            res.append( [1. - p, p] )

        return np.array(res, dtype=float)


    def predict_proba(self, X):
        return self.predict(X)


##def DTWDistance(s1, s2, w):
##    res = ctypes.c_float(0.)
##    DLL.DTWDistance(s1.ctypes.data, s2.ctypes.data, ctypes.c_int(w), ctypes.addressof(res))
##    return res.value





def get_data_matrix(f, key_word):
    mat = sio.loadmat(f)
    key = [k for k in mat.keys() if key_word in k][0]
    data = mat[key]

    freq = data['sampling_frequency'][0][0][0][0]
    sensors = data['data'][0,0].shape[0]
    length = data['data'][0,0].shape[1]
    length_sec = data['data_length_sec'][0][0][0][0]

    return data['data'][0,0].astype(np.float32), freq, sensors, length







def read_data():
    data = []

    data_files = [path_train + f for f in os.listdir(path_train)]

    VEC_LEN = features.READ_LEN

    for fn in data_files:
        with open(fn, "rb") as fin:
            fin.seek(0, 2)
            size = fin.tell() / 8.
            fin.seek(0)
            read = 0
            while read < size:
                a = array.array('d')
                a.fromfile(fin, VEC_LEN)

                data.append(list(a))
                read += VEC_LEN

    return np.array(data, dtype=np.float32)



def train():
    data = read_data()
    print data[:2,:]


    c = [RandomForestClassifier(n_estimators=500)] * 5
    #c = [LogisticRegression()] * 5

##    trees = 20
##    k = 10
##    lnum = 15
##    print "TREES %d, K %d, LNUM %d" % (trees, k, lnum)
##    c = RF(trees, k, lnum)
##    c.alloc()

    if isinstance(c, RF):
        c.fit(data[:,1:].astype(np.float64), data[:,0].astype(np.float64))
    else:
##        c[0].fit(data[:,1:X_LEN], data[:,0])
##        c[1].fit(data[:,X_LEN:2*X_LEN], data[:,0])
##        c[2].fit(data[:,2*X_LEN:3*X_LEN], data[:,0])
##        c[3].fit(data[:,3*X_LEN:4*X_LEN], data[:,0])
##        c[4].fit(data[:,4*X_LEN:5*X_LEN], data[:,0])
        c[0].fit(data[:,1:], data[:,0])
##        c[1].fit(data[:,1:], data[:,0])
##        c[2].fit(data[:,1:], data[:,0])
##        c[3].fit(data[:,1:], data[:,0])
##        c[4].fit(data[:,1:], data[:,0])


    #print "feature importance:"
    #for id, fi in enumerate(c.feature_importances_):
    #    print id, fi
    #print "======"

    return c


def del_predictor(c):
    if isinstance(c, RF):
        c.dealloc()




def process(c):

    y = []
    result = []

    #interictal_files = []

    for i in test_ii_indices:
        fn = files.INTERICTAL_FILES[ i ]
        data_matrix, freq, sensors, length = get_data_matrix(fn, 'interictal_segment')
        vec = []
        for s in range(sensors):
            a = features.extract_features(data_matrix[s])
            vec.append(a.tolist())

        if isinstance(c, RF):
            p = c.predict_proba(vec)
        else:
            vec = np.array(vec)
            p = c[0].predict_proba(vec)
##            p = []
##            p.append( c[0].predict_proba(vec[:,0:X_LEN]).mean(axis=0) )
##            p.append( c[1].predict_proba(vec[:,X_LEN:2*X_LEN]).mean(axis=0) )
##            p.append( c[2].predict_proba(vec[:,2*X_LEN:3*X_LEN]).mean(axis=0) )
##            p.append( c[3].predict_proba(vec[:,3*X_LEN:4*X_LEN]).mean(axis=0) )
##            p.append( c[4].predict_proba(vec[:,4*X_LEN:5*X_LEN]).mean(axis=0) )

        y.append(INTERICTAL_CLS)
        p = np.mean(p, axis=0)
        print fn, p
        p = INTERICTAL_CLS if p[0] > p[1] else PREICTAL_CLS
        #print fn, p
        result.append(p)


    for i in test_pi_indices:
        fn = files.PREICTAL_FILES[ i ]
        data_matrix, freq, sensors, length = get_data_matrix(fn, 'preictal')
        vec = []
        for s in range(sensors):
            a = features.extract_features(data_matrix[s])
            vec.append(a.tolist())

        if isinstance(c, RF):
            p = c.predict_proba(vec)
        else:
            vec = np.array(vec)
            p = c[0].predict_proba(vec)
##            p = []
##            p.append( c[0].predict_proba(vec[:,0:X_LEN]).mean(axis=0) )
##            p.append( c[1].predict_proba(vec[:,X_LEN:2*X_LEN]).mean(axis=0) )
##            p.append( c[2].predict_proba(vec[:,2*X_LEN:3*X_LEN]).mean(axis=0) )
##            p.append( c[3].predict_proba(vec[:,3*X_LEN:4*X_LEN]).mean(axis=0) )
##            p.append( c[4].predict_proba(vec[:,4*X_LEN:5*X_LEN]).mean(axis=0) )

        y.append(PREICTAL_CLS)
        p = np.mean(p, axis=0)
        print fn, p
        p = INTERICTAL_CLS if p[0] > p[1] else PREICTAL_CLS
        #print fn, p
        result.append(p)

    res = classification_report(y, result)
    print res







def subplot(X, Y, rows, cols, num, m, c):
    plt.subplot(rows, cols, num)
    plt.scatter(X, Y , marker=m, c=c)


def plot(data):
    plt.clf()

    ii = data[:,0] == 0
    pp = data[:,0] == 1

    f1 = 1
    f2 = 2

    X1 = data[ii, f1]
    Y1 = data[ii, f2]
    X2 = data[pp, f1]
    Y2 = data[pp, f2]

    X1 -= data[:,f1].mean()
    X2 -= data[:,f1].mean()

    Y1 -= data[:,f2].mean()
    Y2 -= data[:,f2].mean()

    subplot(X1, Y1, 1, 1, 1, 'o', 'b')
    subplot(X2, Y2, 1, 1, 1, 'x', 'r')

##    plt.scatter(X1, Y1 , marker='o', c = 'b')
##    plt.scatter(X2, Y2 , marker='x', c = 'r')

    #plt.scatter(data[ii, 1], data[ii, 2] , marker='o', c = 'b')
    #plt.scatter(data[pp, 1], data[pp, 2] , marker='x', c = 'r')

    plt.show()



def plot_all(data):
    plt.clf()

    ii = data[:,0] == 0
    pp = data[:,0] == 1

    FEATURES = 3

    f1 = 1
    f2 = 2

    X1 = data[ii, f1]
    Y1 = data[ii, f2]
    X2 = data[pp, f1]
    Y2 = data[pp, f2]

    X1 -= data[:,f1].mean()
    X2 -= data[:,f1].mean()

    Y1 -= data[:,f2].mean()
    Y2 -= data[:,f2].mean()

    subplot(X1, Y1, 1, 1, 1, 'o', 'b')
    subplot(X2, Y2, 1, 1, 1, 'x', 'r')

##    plt.scatter(X1, Y1 , marker='o', c = 'b')
##    plt.scatter(X2, Y2 , marker='x', c = 'r')

    #plt.scatter(data[ii, 1], data[ii, 2] , marker='o', c = 'b')
    #plt.scatter(data[pp, 1], data[pp, 2] , marker='x', c = 'r')

    plt.show()



def main():
    c = train()
    process(c)
    del_predictor(c)

    #data = read_data()
    #plot(data)




if __name__ == '__main__':
    main()

