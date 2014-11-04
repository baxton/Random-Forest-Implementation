


import os
import sys
import array
import numpy as np
import scipy as sp
import scipy.io as sio
import ctypes
import datetime as dt
import heapq

import sklearn.hmm as hmm
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import  sklearn.lda as lda

from sklearn.metrics import classification_report

import files
import features



path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_train = path + "prepared_sub5\\"


PREICTAL_CLS = 1
INTERICTAL_CLS = 0

X_LEN = features.READ_LEN


patients = set(files.Patient_1_interictal_files + files.Patient_1_preictal_files + files.Patient_1_test_files +
               files.Patient_2_interictal_files + files.Patient_2_preictal_files + files.Patient_2_test_files)



PATIENT = 1
DOG = 0


FILE_IDX  = 0
SEG_IDX   = 1
SUB_IDX   = 2
CLS_IDX   = 3
X_IDX     = 4


PREP_D = 140
N_COMP_D = 1
NUM_IN_HEAP_D = 2

PREP_P = 180
N_COMP_P = 1
NUM_IN_HEAP_P = 2






def get_segment(fname):
    if "interictal" in fname:
        return "interictal_segment"
    elif "preictal" in fname:
        return "preictal_segment"

    return"test_segment"


def get_cls(segment):
    if "interictal" in segment:
        return INTERICTAL_CLS

    return PREICTAL_CLS






def get_data_matrix(f, key_word):
    mat = sio.loadmat(f)
    key = [k for k in mat.keys() if key_word in k][0]
    data = mat[key]

    freq = data['sampling_frequency'][0][0][0][0]
    sensors = data['data'][0,0].shape[0]
    length = data['data'][0,0].shape[1]
    length_sec = data['data_length_sec'][0][0][0][0]

    return data['data'][0,0], freq, sensors, length







def read_data():
    data = None

    data_files = [path_train + f for f in os.listdir(path_train)]

    length = features.READ_LEN

    for fn in data_files:
        with open(fn, "r") as fin:
            tmp = np.loadtxt(fin, delimiter=',')
            if None == data:
                data = tmp
            else:
                data = sp.concatenate((data, tmp), axis=0)

    return data.astype(np.float32)







def prep_data(data, length):
    rows = data.shape[0]
    cols = data.shape[1]

    step = int(length * .1)

    result = []

    for id, d in enumerate(data):
        beg = 0
        end = beg + length

        while end < cols:
            result.append(d[beg:end])

            beg = beg + step  #(beg + end) / 2
            end = beg + length
        result.append(d[-length:])

    return np.array(result, dtype=np.float32)










def process():
    data = read_data()
    print "# Data loaded"

    rows = data.shape[0]
    cols = data.shape[1]


    iid = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == DOG)
    ipd = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == DOG)

    iip = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)
    ipp = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)


    did = prep_data(data[iid, X_IDX:], PREP_D)
    dip = prep_data(data[iip, X_IDX:], PREP_P)
    dpd = prep_data(data[ipd, X_IDX:], PREP_D)
    dpp = prep_data(data[ipp, X_IDX:], PREP_P)

    print "did", did.shape
    print "dip", dip.shape
    print "dpd", dpd.shape
    print "dpp", dpp.shape


    cid = hmm.GMMHMM(n_components = N_COMP_D, n_iter = 1000)
    cip = hmm.GMMHMM(n_components = N_COMP_P, n_iter = 1000)
    cpd = hmm.GMMHMM(n_components = N_COMP_D, n_iter = 1000)
    cpp = hmm.GMMHMM(n_components = N_COMP_P, n_iter = 1000)

    cid.fit([did])
    cip.fit([dip])
    cpd.fit([dpd])
    cpp.fit([dpp])
    print "Fitting finished"

    cid_prob = sp.log( did.shape[0] / float(did.shape[0] + dpd.shape[0]) )
    cpd_prob = sp.log( dpd.shape[0] / float(did.shape[0] + dpd.shape[0]) )

    cip_prob = sp.log( dip.shape[0] / float(dip.shape[0] + dpp.shape[0]) )
    cpp_prob = sp.log( dpp.shape[0] / float(dip.shape[0] + dpp.shape[0]) )

    print "cid_prob", cid_prob
    print "cpd_prob", cpd_prob
    print "cip_prob", cip_prob
    print "cpp_prob", cpp_prob


    dog_vs_man = RandomForestClassifier(n_estimators=2000)
    dog_vs_man.fit(data[:,X_IDX:], data[:,SUB_IDX])

    # free some memory
    data = None
    did = None
    dip = None
    dpd = None
    dpp = None

    ##########################################

    for i, fn in files.TEST_FILES.items():

        seg = get_segment(fn)
        cls = get_cls(seg)

        data_matrix, freq, sensors, length = get_data_matrix(fn, seg)

        p = 0.
        cnt = 0.

        for s in range(sensors):
            vec = features.extract_features( data_matrix[s] )

            d_or_m = dog_vs_man.predict(vec)

            vec = prep_data(vec.reshape((1, vec.shape[0])), (PREP_P if d_or_m == PATIENT else PREP_D) )

            h = []

            for v in vec:
                if d_or_m == PATIENT:
                    pi = cip.score([v]) + cip_prob
                    pp = cpp.score([v]) + cpp_prob
                else:
                    pi = cid.score([v]) + cid_prob
                    pp = cpd.score([v]) + cpd_prob


                delta = abs(pi - pp)
                #print "#>>", "%2.20f vs %2.20f (%f)" % (pi, pp, delta)
                heapq.heappush(h, (delta, pi, pp))

            num_items = NUM_IN_HEAP_P if d_or_m == PATIENT else NUM_IN_HEAP_D
            t = 0.
            for delta, pi, pp in heapq.nlargest(num_items, h):
                t += float(INTERICTAL_CLS if pi > pp else PREICTAL_CLS)
            t /= float(num_items)
            #print "# sensor", s, "max N:", heapq.nlargest(num_items+1, h)

            p += float(INTERICTAL_CLS if t < .5 else PREICTAL_CLS)
            cnt += 1.

        if cnt != 0.:
            p /= cnt
            v = INTERICTAL_CLS if p < .5 else PREICTAL_CLS
            print "# %s,%d (%f)" % (fn, v, p)
        else:
            p = 1.
            v = 1.
            print "# %s,%d (%f)" % (fn, v, p)







def main():
    sp.random.seed()
    process()



if __name__ == '__main__':
    main()



