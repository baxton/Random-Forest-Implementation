

import numpy as np
import scipy as sp
import ctypes


RF_DLL = ctypes.cdll.LoadLibrary("C:\\Temp\\kaggle\\epilepsy\\scripts\\sub5\\rf.dll")


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
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)

        rows = X.shape[0]
        columns = X.shape[1]
        RF_DLL.fit_rf(X.ctypes.data, Y.ctypes.data, ctypes.c_int(rows), ctypes.c_int(columns))

    def predict(self, X):
        X = np.array(X, dtype=float)

        res = []

        if len(X.shape) == 2:
            rows = X.shape[0]
            columns = X.shape[1]

            for r in range(rows):
                v = X[r, :]
                p = ctypes.c_double(0.)
                RF_DLL.predict_rf(v.ctypes.data, ctypes.c_int(columns), ctypes.addressof(p))

                p = p.value
                res.append( [1. - p, p] )
        else:
            columns = X.shape[0]
            v = X
            p = ctypes.c_double(0.)
            RF_DLL.predict_rf(v.ctypes.data, ctypes.c_int(columns), ctypes.addressof(p))

            p = p.value
            res.append( [1. - p, p] )


        return np.array(res, dtype=float)


    def predict_proba(self, X):
        return self.predict(X)
# END OF RF

