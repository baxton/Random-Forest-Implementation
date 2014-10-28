

import numpy as np
import scipy as sp
import numpy.fft as fft



X_LEN = 600
MA = X_LEN / 10

READ_LEN = 1 + 1 + 1 + (X_LEN - MA + 1 - MA + 1)




def moving_average(a, n=3) :
    if 1 == len(a.shape):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    else:
        ret = np.cumsum(a, dtype=float, axis=1)
        ret[:,n:] = ret[:,n:] - ret[:,:-n]
        return ret[:,n - 1:] / n



def Z_normalize(vec):
    m = vec.mean()
    #s = vec.std()
    s = max( vec.max(), np.abs(vec.min()) )

    vec -= m
    vec /= s

    return vec



def PAA(ORIG, bins):
    N = ORIG.shape[0]
    bags = bins
    bag_size = N / bags

    vec = sp.zeros((bags,), dtype=np.float32)

    for i in range(bags-1):
        begin = i * bag_size
        end = begin + bag_size
        vec[i] = ORIG[begin:end].mean()

    vec[-1] = ORIG[end:].mean()

    return vec






def extract_features(ts):

    ts = ts.astype(np.float32)

    features1 = Z_normalize(ts)
    features1 = PAA(features1, X_LEN)
    features1 = moving_average(features1, MA)
    features1 = moving_average(features1, MA)
    ret = features1.astype(np.float32)


    return ret
