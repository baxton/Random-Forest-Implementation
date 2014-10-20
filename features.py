

import numpy as np
import scipy as sp
import numpy.fft as fft


X_LEN = 4
#READ_LEN = 1 + X_LEN*5
READ_LEN = 1 + 4





def PAA(ORIG, bins):
    N = ORIG.shape[0]
    bags = bins
    bag_size = N / bags

    vec = sp.zeros((bags,), dtype=np.float32)

    for i in range(bags):
        begin = i * bag_size
        end = begin + bag_size if begin + bag_size < N else N
        vec[i] = ORIG[begin:end].mean()

    return vec



def PA_max(ORIG, bins):
    N = ORIG.shape[0]
    bags = bins
    bag_size = N / bags

    vec = sp.zeros((bags,), dtype=np.float32)

    for i in range(bags):
        begin = i * bag_size
        end = begin + bag_size if begin + bag_size < N else N
        vec[i] = ORIG[begin:end].max()

    return vec


def PA_min(ORIG, bins):
    N = ORIG.shape[0]
    bags = bins
    bag_size = N / bags

    vec = sp.zeros((bags,), dtype=np.float32)

    for i in range(bags):
        begin = i * bag_size
        end = begin + bag_size if begin + bag_size < N else N
        vec[i] = ORIG[begin:end].min()

    return vec




def PA_integral(ORIG, bins):
    N = ORIG.shape[0]
    bags = bins
    bag_size = N / bags

    vec = sp.zeros((bags,), dtype=np.float32)

    for i in range(bags):
        begin = i * bag_size
        end = begin + bag_size if begin + bag_size < N else N
        vec[i] = ORIG[begin:end].sum()

    return vec






def extract_features(ts):
    #ts -= ts.mean()

##    tmp = ts[0:-1] - ts[1:]
##    tmp = tmp[0:-1] - tmp[1:]
##
##    features1 = PAA(tmp, X_LEN)
##
##    features2 = fft.fft(ts, X_LEN).astype(np.float32)
##    features3 = features1 ** 2

    features4 = PA_integral(ts, X_LEN)

##    features5 = PA_max(ts, X_LEN)


    #return sp.concatenate((features1, features2, features3))
    #return features1

    #ret = sp.concatenate((features1, features2, features3, features4, features5))
    ret = np.array([features4.mean(), features4.max(), features4.min(), features4.std()], dtype=np.float64)

    if READ_LEN != (ret.shape[0] + 1):
        raise Exception("READ_LEN is not adjusted %d vs %d" % (READ_LEN, ret.shape[0]+1))

    if X_LEN != ret.shape[0]:
        raise Exception("X_LEN is not adjusted %d vs %d" % (X_LEN, ret.shape[0]))

    return ret
