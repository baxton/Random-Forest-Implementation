

import numpy as np
import scipy as sp
import numpy.fft as fft
from sklearn import preprocessing


X_LEN = 1200
MA = X_LEN / 10

READ_LEN = 1 + 1 + 1 + X_LEN




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
    s = vec.std()
    #s = max( vec.max(), np.abs(vec.min()) )

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






def extract_features(ts, start=5000, step=1000):

    ts = ts.astype(np.float32)

    D = 0 if (1 == len(ts.shape)) else 1

    num = start

    trend = moving_average(ts, num)
    num -= step
    trend = moving_average(trend, num)
    num -= step
    trend = moving_average(trend, num)
    num -= step
    trend = moving_average(trend, num)
    num -= step
    trend = moving_average(trend, num)

    L = min(ts.shape[D], trend.shape[D])

    if 0 == D:
        sesonal = ts[:L] - trend[:L]
    else:
        sesonal = ts[:,:L] - trend[:,:L]
    sesonal = moving_average(sesonal, num)

    L = min(ts.shape[D], trend.shape[D], sesonal.shape[D])
    if 0 == D:
        noise = (ts[:L] - trend[:L]) - sesonal[:L]
    else:
        noise = (ts[:,:L] - trend[:,:L]) - sesonal[:,:L]

    return trend, sesonal, noise





