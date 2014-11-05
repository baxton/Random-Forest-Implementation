
import os
import numpy.fft as fft
import numpy as np
import scipy as sp
import scipy.spatial.distance as spd

import sklearn.hmm as hmm
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
import ctypes
import scipy.io as sio
import datetime as dt

import heapq

from sklearn.metrics import classification_report

import files
import features



DLL = ctypes.cdll.LoadLibrary("C:\\Temp\\kaggle\\epilepsy\\scripts\\sub5\\dtw_fast.dll")
#DLL = ctypes.cdll.LoadLibrary("/home/maxim/kaggle/epilepsy/scripts/sub2/dtw_fast.dll")


#path = "/home/maxim/kaggle/epilepsy/data/"
#path_train = path + "prepared_sub2/"
path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_train = path + "prepared_sub5\\"


patients = set(files.Patient_1_interictal_files + files.Patient_1_preictal_files + files.Patient_1_test_files +
               files.Patient_2_interictal_files + files.Patient_2_preictal_files + files.Patient_2_test_files)


PREICTAL_CLS = 1
INTERICTAL_CLS = 0


PATIENT = 1
DOG = 0


FILE_IDX  = 0
SEG_IDX   = 1
SUB_IDX   = 2
CLS_IDX   = 3
X_IDX     = 4


W = 600

PREP = 180
N_COMP = 1
NUM_IN_HEAP = 11

PREP_D = 140
N_COMP_D = 1
NUM_IN_HEAP_D = 7




def DTWDistance(s1, s2, w, giveup_threshold = float('inf')):
    res = ctypes.c_float(0.)
    code = DLL.DTWDistance(s1.ctypes.data, s2.ctypes.data, ctypes.c_int(s1.shape[0]), ctypes.c_int(s2.shape[0]), ctypes.c_int(w), ctypes.c_float(giveup_threshold), ctypes.addressof(res))
    return res.value



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

    data_files = [path_train + f for f in os.listdir(path_train) if 'fast' not in f]

    length = features.READ_LEN

    for fn in data_files:
        with open(fn, "r") as fin:
            tmp = np.loadtxt(fin, delimiter=',')
            if None == data:
                data = tmp
            else:
                data = sp.concatenate((data, tmp), axis=0)

    return data.astype(np.float32)



def process():
    data = read_data()

    LEN = features.X_LEN

    rows = data.shape[0]
    cols = data.shape[1]

    iid = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == DOG)
    ipd = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == DOG)

    iip = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)
    ipp = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)


    # prepare mean
    mean = sp.zeros((LEN,), dtype=np.float32)

    mean = np.mean(data[iid | iip,X_IDX:X_IDX+LEN], axis=0)

    #fft_p = fft.fft(data[0,X_IDX:X_IDX+LEN]).astype(np.float32)
    fft_p = np.mean(data[ipp,X_IDX:X_IDX+LEN], axis=0)


    rand_p = sp.random.rand(LEN).astype(np.float32)
    orig_p = sp.zeros(LEN).astype(np.float32)
    p_5000 = sp.zeros(LEN).astype(np.float32)
    p_5000 += 5000

    print mean[:5]


    # get distances

    dist_mean = sp.zeros((rows,))
    dist_fft = sp.zeros((rows,))
    dist_rnd = sp.zeros((rows,))
    dist_orig = sp.zeros((rows,))
    dist_5000 = sp.zeros((rows,))

    for i in range(rows):
    #for i in range(5):
        #dist_mean[i] = DTWDistance(mean, data[i,X_IDX:X_IDX+LEN], W)
        dist_mean[i] = DTWDistance(mean, data[i,X_IDX:X_IDX+LEN], W)
        #dist_mean[i] = spd.cosine(mean, data[i,X_IDX:X_IDX+LEN])
        dist_fft[i] = DTWDistance(fft_p, data[i,X_IDX:X_IDX+LEN], W)
        #dist_fft[i] = spd.cosine(fft_p, data[i,X_IDX:])
        #dist_rnd[i] = DTWDistance(rand_p, data[i,X_IDX:], W)
        #dist_orig[i] = DTWDistance(orig_p, data[i,X_IDX:X_IDX+LEN], W)
        #dist_orig[i] = spd.cosine(orig_p, data[i,X_IDX:X_IDX+LEN])
        #dist_5000[i] = DTWDistance(p_5000, data[i,X_IDX:X_IDX+LEN], W)




    plt.clf()

    plt.subplot(411)
    plt.scatter(dist_fft[iid], dist_mean[iid], c='r', marker='o')
    #plt.scatter(dist_mean[ip], dist_fft[ip], c='g', marker='x')

    plt.subplot(412)
    #plt.scatter(dist_mean[ii], dist_rnd[ii], c='r', marker='o')
    plt.scatter(dist_fft[iip], dist_mean[iip], c='g', marker='x')

    plt.subplot(413)
    plt.scatter(dist_fft[ipd], dist_mean[ipd], c='r', marker='o')
    #plt.scatter(dist_mean[ip], dist_fft[ip], c='g', marker='x')

    plt.subplot(414)
    plt.scatter(dist_fft[ipp], dist_mean[ipp], c='g', marker='o')
    #plt.scatter(dist_orig[ip], dist_fft[ip], c='g', marker='x')


    plt.show()


    #plt.clf()
    #x = np.array(range(10), dtype=float)
    #y = x ** 2
    #plt.plot(x, y)
    #plt.show()






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




def process_hmm(data):

    test_ii_indices = [424,295,67,340,107,231,134,443,268,171,86,391,220,267,236,345,388,232,88,57,691,800,840,967,509,582,521,620,488,742,755,740,492,577,637,781,496,787,958,782,1874,981,1197,2401,1914,1798,2312,2348,1950,1090,2034,1512,1875,1604,1867,1441,1298,1772,1842,2061,2990,3008,2728,3188,3160,2507,3070,2513,2836,2520,2430,2677,2999,2943,3028,3104,2977,3165,3209,2530,3626,3616,3427,3275,3396,3229,3497,3649,3424,3547,3365,3496,3652,3512,3537,3267,3542,3369,3262,3611,3719,3675,3676,3712,3721,3682,3683,3684,3687,3688,3717,3691,3723,3693,3711,3695,3696,3698,3702,3706,3725,3726,3727,3731,3734,3737,3739,3740,3741,3742,3743,3745,3746,3747,3765,3751,3753,3764,3759,3763]
    test_pi_indices = [3784,3767,3769,3773,3787,3775,3788,3780,3782,3783,3812,3791,3792,3816,3808,3826,3810,3813,3814,3803,3883,3895,3872,3900,3843,3846,3838,3839,3861,3881,3953,3946,3954,3951,3964,3931,3938,3936,3927,3999,4002,4003,4029,4015,4030,4008,4017,4010,4014,4019,4032,4034,4038,4039,4040,4044,4045,4048,4051,4054,4057,4058,4059,4060,4061,4064]


    ###############################################################3
    #data = read_data()

    LEN = features.X_LEN

    rows = data.shape[0]
    cols = data.shape[1]


    iid = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == DOG)
    ipd = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == DOG)

    iip = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)
    ipp = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)


    did = prep_data(data[iid, X_IDX:], PREP)
    dip = prep_data(data[iip, X_IDX:], PREP)
    dpd = prep_data(data[ipd, X_IDX:], PREP)
    dpp = prep_data(data[ipp, X_IDX:], PREP)

    print "did", did.shape
    print "dip", dip.shape
    print "dpd", dpd.shape
    print "dpp", dpp.shape


    cid = hmm.GMMHMM(n_components = N_COMP)  #, n_iter = 500)
    cip = hmm.GMMHMM(n_components = N_COMP)  #, n_iter = 500)
    cpd = hmm.GMMHMM(n_components = N_COMP)  #, n_iter = 500)
    cpp = hmm.GMMHMM(n_components = N_COMP)  #, n_iter = 500)

    cid.fit([did])
    #cip.fit([dip])
    cpd.fit([dpd])
    #cpp.fit([dpp])
    print "Fitting finished"

    cid_prob = sp.log( did.shape[0] / float(did.shape[0] + dpd.shape[0]) )
    cpd_prob = sp.log( dpd.shape[0] / float(did.shape[0] + dpd.shape[0]) )

    cip_prob = sp.log( dip.shape[0] / float(dip.shape[0] + dpp.shape[0]) )
    cpp_prob = sp.log( dpp.shape[0] / float(dip.shape[0] + dpp.shape[0]) )

    print "cid_prob", cid_prob
    print "cpd_prob", cpd_prob
    print "cip_prob", cip_prob
    print "cpp_prob", cpp_prob



    y = []
    result = []

    for i in (test_ii_indices + test_pi_indices):

        if i in patients:
            continue

        if i in files.INTERICTAL_FILES:
            fn = files.INTERICTAL_FILES[ i ]
        else:
            fn = files.PREICTAL_FILES[ i ]

        seg = get_segment(fn)
        cls = get_cls(seg)

        y.append(cls)



        data_matrix, freq, sensors, length = get_data_matrix(fn, seg)

        p = 0.
        cnt = 0.
        for s in range(sensors):
            vec = features.extract_features( data_matrix[s] )

            #dist = DTWDistance(mean_p, vec[X_IDX:], W)

            vec = prep_data(vec.reshape((1, vec.shape[0])), PREP)

            h = []

            for v in vec:
                if False:
##                    pi = cip.score([v]) + cip_prob
##                    pp = cpp.score([v]) + cpp_prob
                    pi = cip.score([v])
                    pp = cpp.score([v])
                else:
##                    pi = cid.score([v]) + cid_prob
##                    pp = cpd.score([v]) + cpd_prob
                    pi = cid.score([v]) + cid_prob
                    pp = cpd.score([v]) + cpd_prob


                delta = abs(pi - pp)
                #print "#>>", "%2.20f vs %2.20f (%f)" % (pi, pp, delta)
                heapq.heappush(h, (delta, pi, pp))

            num_items = NUM_IN_HEAP
            t = 0.
            for delta, pi, pp in heapq.nlargest(num_items, h):
                t += float(INTERICTAL_CLS if pi > pp else PREICTAL_CLS)
            t /= float(num_items)
            print "# sensor", s, "max N:", heapq.nlargest(num_items+1, h)

            p += INTERICTAL_CLS if t < .5 else PREICTAL_CLS
            cnt += 1.

        if cnt != 0.:
            p /= cnt
            if False:
                v = INTERICTAL_CLS if p < .5 else PREICTAL_CLS
                result.append(v)
                print "# %s,%d (%f)" % (fn, v, p)
            else:
                v = INTERICTAL_CLS if p < .5 else PREICTAL_CLS
                result.append(v)
                print "# %s,%d (%f)" % (fn, v, p)
        else:
            p = 1.
            v = 1.
            result.append(v)
            print "# %s,%d (%f)" % (fn, v, p)

    res = classification_report(y, result)
    print res




def process_hmm2():

    test_ii_indices = [424,295,67,340,107,231,134,443,268,171,86,391,220,267,236,345,388,232,88,57,691,800,840,967,509,582,521,620,488,742,755,740,492,577,637,781,496,787,958,782,1874,981,1197,2401,1914,1798,2312,2348,1950,1090,2034,1512,1875,1604,1867,1441,1298,1772,1842,2061,2990,3008,2728,3188,3160,2507,3070,2513,2836,2520,2430,2677,2999,2943,3028,3104,2977,3165,3209,2530,3626,3616,3427,3275,3396,3229,3497,3649,3424,3547,3365,3496,3652,3512,3537,3267,3542,3369,3262,3611,3719,3675,3676,3712,3721,3682,3683,3684,3687,3688,3717,3691,3723,3693,3711,3695,3696,3698,3702,3706,3725,3726,3727,3731,3734,3737,3739,3740,3741,3742,3743,3745,3746,3747,3765,3751,3753,3764,3759,3763]
    test_pi_indices = [3784,3767,3769,3773,3787,3775,3788,3780,3782,3783,3812,3791,3792,3816,3808,3826,3810,3813,3814,3803,3883,3895,3872,3900,3843,3846,3838,3839,3861,3881,3953,3946,3954,3951,3964,3931,3938,3936,3927,3999,4002,4003,4029,4015,4030,4008,4017,4010,4014,4019,4032,4034,4038,4039,4040,4044,4045,4048,4051,4054,4057,4058,4059,4060,4061,4064]


    ###############################################################3
    data = read_data()

    LEN = features.X_LEN

    rows = data.shape[0]
    cols = data.shape[1]

    Y = sp.zeros((rows,), dtype=int)
    i = 0
    for d in data:
        if d[FILE_IDX] in patients:
            Y[i] = PATIENT
        else:
            Y[i] = DOG
        i += 1


    ii = (data[:,CLS_IDX] == INTERICTAL_CLS)
    ip = (data[:,CLS_IDX] == PREICTAL_CLS)


    di, di_rows = prep_data(data[ii, X_IDX:], PREP)
    dp, dp_rows = prep_data(data[ip, X_IDX:], PREP)

    print di.shape
    print dp.shape


    ci = hmm.GMMHMM(n_components = N_COMP, n_iter = 500)
    cp = hmm.GMMHMM(n_components = N_COMP, n_iter = 500)

    ci.fit([di])
    cp.fit([dp])
    print "Fitting finished"

    ci_prob = sp.log( di_rows / float(di_rows + dp_rows) )
    cp_prob = sp.log( dp_rows / float(di_rows + dp_rows) )


    #mean_p = np.mean(data[ipp,X_IDX:], axis=0)



    y = []
    result = []

    for i in (test_ii_indices + test_pi_indices):
        if i in patients:
            continue

        if i in files.INTERICTAL_FILES:
            fn = files.INTERICTAL_FILES[ i ]
        else:
            fn = files.PREICTAL_FILES[ i ]

        seg = get_segment(fn)
        cls = get_cls(seg)

        y.append(cls)



        data_matrix, freq, sensors, length = get_data_matrix(fn, seg)

        p = 0.
        cnt = 0.
        for s in range(sensors):
            vec = features.extract_features( data_matrix[s] )

            #dist = DTWDistance(mean_p, vec[X_IDX:], W)

            vec, r = prep_data(vec.reshape((1, vec.shape[0])), PREP)

            for v in vec:
                pi = ci.score([v]) + ci_prob
                pp = cp.score([v]) + cp_prob

                print "#>>", "%2.20f vs %2.20f" % (pi, pp)
                p += INTERICTAL_CLS if pi > pp else PREICTAL_CLS
                cnt += 1.

        p /= cnt
        v = INTERICTAL_CLS if p < .5 else PREICTAL_CLS
        result.append(v)
        print "# %s,%d (%f)" % (fn, v, p)

    res = classification_report(y, result)
    print res




def dog_vs_man():
    test_ii_indices = [424,295,67,340,107,231,134,443,268,171,86,391,220,267,236,345,388,232,88,57,691,800,840,967,509,582,521,620,488,742,755,740,492,577,637,781,496,787,958,782,1874,981,1197,2401,1914,1798,2312,2348,1950,1090,2034,1512,1875,1604,1867,1441,1298,1772,1842,2061,2990,3008,2728,3188,3160,2507,3070,2513,2836,2520,2430,2677,2999,2943,3028,3104,2977,3165,3209,2530,3626,3616,3427,3275,3396,3229,3497,3649,3424,3547,3365,3496,3652,3512,3537,3267,3542,3369,3262,3611,3719,3675,3676,3712,3721,3682,3683,3684,3687,3688,3717,3691,3723,3693,3711,3695,3696,3698,3702,3706,3725,3726,3727,3731,3734,3737,3739,3740,3741,3742,3743,3745,3746,3747,3765,3751,3753,3764,3759,3763]
    test_pi_indices = [3784,3767,3769,3773,3787,3775,3788,3780,3782,3783,3812,3791,3792,3816,3808,3826,3810,3813,3814,3803,3883,3895,3872,3900,3843,3846,3838,3839,3861,3881,3953,3946,3954,3951,3964,3931,3938,3936,3927,3999,4002,4003,4029,4015,4030,4008,4017,4010,4014,4019,4032,4034,4038,4039,4040,4044,4045,4048,4051,4054,4057,4058,4059,4060,4061,4064]


    ###############################################################
    data = read_data()

    print data.shape
    print data[0,:6]

    LEN = features.X_LEN

    rows = data.shape[0]
    cols = data.shape[1]

    Y = sp.zeros((rows,1))
    i = 0
    for d in data:
        if d[FILE_IDX] in patients:
            Y[i] = PATIENT
        else:
            Y[i] = DOG
        i += 1

#    iid = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == DOG)
#    ipd = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == DOG)

#    iip = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)
#    ipp = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)


    ###############################################################
    c = svm.SVC()
    c.fit(data[:,X_IDX:], Y)

    y = []
    result = []

    for i in (test_ii_indices + test_pi_indices):
        #if i in patients:
        #    continue

        if i in files.INTERICTAL_FILES:
            fn = files.INTERICTAL_FILES[ i ]
        else:
            fn = files.PREICTAL_FILES[ i ]

        seg = get_segment(fn)
        cls = get_cls(seg)

        if i in patients:
            y.append(PATIENT)
        else:
            y.append(DOG)

        data_matrix, freq, sensors, length = get_data_matrix(fn, seg)

        p = 0.
        cnt = 0.
        for s in range(sensors):
            vec = features.extract_features( data_matrix[s] )
            p += c.predict(vec)
            cnt += 1.

        p /= cnt
        result.append(DOG if p == 0. else PATIENT)
        print "# %s,%f (%s)" % (fn, p, "DOG" if p == 0. else "MAN")

    res = classification_report(y, result)
    print res




def dog_vs_man2():
    test_ii_indices = [424,295,67,340,107,231,134,443,268,171,86,391,220,267,236,345,388,232,88,57,691,800,840,967,509,582,521,620,488,742,755,740,492,577,637,781,496,787,958,782,1874,981,1197,2401,1914,1798,2312,2348,1950,1090,2034,1512,1875,1604,1867,1441,1298,1772,1842,2061,2990,3008,2728,3188,3160,2507,3070,2513,2836,2520,2430,2677,2999,2943,3028,3104,2977,3165,3209,2530,3626,3616,3427,3275,3396,3229,3497,3649,3424,3547,3365,3496,3652,3512,3537,3267,3542,3369,3262,3611,3719,3675,3676,3712,3721,3682,3683,3684,3687,3688,3717,3691,3723,3693,3711,3695,3696,3698,3702,3706,3725,3726,3727,3731,3734,3737,3739,3740,3741,3742,3743,3745,3746,3747,3765,3751,3753,3764,3759,3763]
    test_pi_indices = [3784,3767,3769,3773,3787,3775,3788,3780,3782,3783,3812,3791,3792,3816,3808,3826,3810,3813,3814,3803,3883,3895,3872,3900,3843,3846,3838,3839,3861,3881,3953,3946,3954,3951,3964,3931,3938,3936,3927,3999,4002,4003,4029,4015,4030,4008,4017,4010,4014,4019,4032,4034,4038,4039,4040,4044,4045,4048,4051,4054,4057,4058,4059,4060,4061,4064]


    ###############################################################
    data = read_data()

    print data.shape
    print data[0,:6]

    LEN = features.X_LEN

    rows = data.shape[0]
    cols = data.shape[1]

    Y = sp.zeros((rows,1))
    i = 0
    for d in data:
        if d[FILE_IDX] in patients:
            Y[i] = PATIENT
        else:
            Y[i] = DOG
        i += 1

#    iid = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == DOG)
#    ipd = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == DOG)

#    iip = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)
#    ipp = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)


    ###############################################################
    c = RandomForestClassifier(n_estimators=1000)
    c.fit(data[:,X_IDX:], Y)

    y = []
    result = []

    for i in (test_ii_indices + test_pi_indices):
        #if i in patients:
        #    continue

        if i in files.INTERICTAL_FILES:
            fn = files.INTERICTAL_FILES[ i ]
        else:
            fn = files.PREICTAL_FILES[ i ]

        seg = get_segment(fn)
        cls = get_cls(seg)

        if i in patients:
            y.append(PATIENT)
        else:
            y.append(DOG)

        data_matrix, freq, sensors, length = get_data_matrix(fn, seg)

        p = 0.
        cnt = 0.
        for s in range(sensors):
            vec = features.extract_features( data_matrix[s] )
            t = c.predict(vec)
            print "# sensor", s, "class", t
            p += t[0]
            cnt += 1.

        p /= cnt
        result.append(DOG if p == 0. else PATIENT)
        print "# %s,%f (%s)" % (fn, p, "DOG" if p == 0. else "MAN")

    res = classification_report(y, result)
    print res




def write(fout, attribs, vectors, orig):
    for attr in attribs:
        pi = attr[1]
        pp = attr[2]
        id = attr[3]

        to_file = None

        if pi > pp:
            # interictal
            if orig[SUB_IDX] == PATIENT:
                to_file = sp.concatenate(([orig[FILE_IDX], orig[SEG_IDX], PATIENT, INTERICTAL_CLS], vectors[id]))
            else:
                to_file = sp.concatenate(([orig[FILE_IDX], orig[SEG_IDX], DOG, INTERICTAL_CLS], vectors[id]))
        else:
            # preictal
            if orig[SUB_IDX] == PATIENT:
                to_file = sp.concatenate(([orig[FILE_IDX], orig[SEG_IDX], PATIENT, PREICTAL_CLS], vectors[id]))
            else:
                to_file = sp.concatenate(([orig[FILE_IDX], orig[SEG_IDX], DOG, PREICTAL_CLS], vectors[id]))

        to_file.tofile(fout, sep=',')
        fout.write(os.linesep)
        fout.flush()



def gen_train_sets():
    data = read_data()

    LEN = features.X_LEN

    rows = data.shape[0]
    cols = data.shape[1]


    iid = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == DOG)
    ipd = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == DOG)

    iip = (data[:,CLS_IDX] == INTERICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)
    ipp = (data[:,CLS_IDX] == PREICTAL_CLS) & (data[:,SUB_IDX] == PATIENT)


    did = prep_data(data[iid, X_IDX:], PREP_D)
    dip = prep_data(data[iip, X_IDX:], PREP)
    dpd = prep_data(data[ipd, X_IDX:], PREP_D)
    dpp = prep_data(data[ipp, X_IDX:], PREP)

    print "did", did.shape
    print "dip", dip.shape
    print "dpd", dpd.shape
    print "dpp", dpp.shape


    cid = hmm.GMMHMM(n_components = N_COMP_D, n_iter = 500)
    cip = hmm.GMMHMM(n_components = N_COMP, n_iter = 500)
    cpd = hmm.GMMHMM(n_components = N_COMP_D, n_iter = 500)
    cpp = hmm.GMMHMM(n_components = N_COMP, n_iter = 500)

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


    did = None
    dip = None
    dpd = None
    dpp = None

    train_fn_d = path_train + "train_fast_d.csv"
    train_fn_p = path_train + "train_fast_p.csv"
    with open(train_fn_d, "w+") as fout_d:
        with open(train_fn_p, "w+") as fout_p:
            for d in data:
                prep = PREP if d[SUB_IDX] == PATIENT else PREP_D

                vec = prep_data(d[X_IDX:].reshape((1, d[X_IDX:].shape[0])), prep)

                h = []

                for id, v in enumerate(vec):
                    if d[SUB_IDX] == PATIENT:
                        pi = cip.score([v]) + cip_prob
                        pp = cpp.score([v]) + cpp_prob
                    else:
                        pi = cid.score([v]) + cid_prob
                        pp = cpd.score([v]) + cpd_prob

                    delta = abs(pi - pp)
                    heapq.heappush(h, (delta, pi, pp, id))

                num_items = NUM_IN_HEAP if d[SUB_IDX] == PATIENT else NUM_IN_HEAP_D
                fout = fout_p if d[SUB_IDX] == PATIENT else fout_d
                write(fout, heapq.nlargest(num_items, h), vec, d)





def read_data_fast():
    data_d = None
    data_p = None

    fn_d = path_train + "train_fast_d.csv"
    fn_p = path_train + "train_fast_p.csv"

    length = features.READ_LEN

    with open(fn_d, "r") as fin:
        data_d = np.loadtxt(fin, delimiter=',')

    with open(fn_p, "r") as fin:
        data_p = np.loadtxt(fin, delimiter=',')

    return data_p.astype(np.float32), data_d.astype(np.float32)



def process_hmm3():

    data_p, data_d = read_data_fast()

    test_ii_indices = [424,295,67,340,107,231,134,443,268,171,86,391,220,267,236,345,388,232,88,57,691,800,840,967,509,582,521,620,488,742,755,740,492,577,637,781,496,787,958,782,1874,981,1197,2401,1914,1798,2312,2348,1950,1090,2034,1512,1875,1604,1867,1441,1298,1772,1842,2061,2990,3008,2728,3188,3160,2507,3070,2513,2836,2520,2430,2677,2999,2943,3028,3104,2977,3165,3209,2530,3626,3616,3427,3275,3396,3229,3497,3649,3424,3547,3365,3496,3652,3512,3537,3267,3542,3369,3262,3611,3719,3675,3676,3712,3721,3682,3683,3684,3687,3688,3717,3691,3723,3693,3711,3695,3696,3698,3702,3706,3725,3726,3727,3731,3734,3737,3739,3740,3741,3742,3743,3745,3746,3747,3765,3751,3753,3764,3759,3763]
    test_pi_indices = [3784,3767,3769,3773,3787,3775,3788,3780,3782,3783,3812,3791,3792,3816,3808,3826,3810,3813,3814,3803,3883,3895,3872,3900,3843,3846,3838,3839,3861,3881,3953,3946,3954,3951,3964,3931,3938,3936,3927,3999,4002,4003,4029,4015,4030,4008,4017,4010,4014,4019,4032,4034,4038,4039,4040,4044,4045,4048,4051,4054,4057,4058,4059,4060,4061,4064]


    ###############################################################3

    LEN = features.X_LEN

    rows_p = data_p.shape[0]
    cols_p = data_p.shape[1]

    rows_d = data_d.shape[0]
    cols_d = data_d.shape[1]


    iid = data_d[:,CLS_IDX] == INTERICTAL_CLS
    ipd = data_d[:,CLS_IDX] == PREICTAL_CLS

    iip = data_p[:,CLS_IDX] == INTERICTAL_CLS
    ipp = data_p[:,CLS_IDX] == PREICTAL_CLS


    did = data_d[iid, X_IDX:]
    dpd = data_d[ipd, X_IDX:]

    dip = data_p[iip, X_IDX:]
    dpp = data_p[ipp, X_IDX:]

    print "did", did.shape
    print "dip", dip.shape
    print "dpd", dpd.shape
    print "dpp", dpp.shape


    cid = hmm.GMMHMM(n_components = N_COMP_D, n_iter = 500)
    cpd = hmm.GMMHMM(n_components = N_COMP_D, n_iter = 500)

    cip = hmm.GMMHMM(n_components = N_COMP, n_iter = 500)
    cpp = hmm.GMMHMM(n_components = N_COMP, n_iter = 500)

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



    y = []
    result = []

    for i in (test_ii_indices + test_pi_indices):

##        if i in patients:
##            continue

        if i in files.INTERICTAL_FILES:
            fn = files.INTERICTAL_FILES[ i ]
        else:
            fn = files.PREICTAL_FILES[ i ]

        seg = get_segment(fn)
        cls = get_cls(seg)

        if i in patients:
            d_or_m = PATIENT
        else:
            d_or_m = DOG

        y.append(cls)



        data_matrix, freq, sensors, length = get_data_matrix(fn, seg)

        heap_sensors = []

        p = 0.
        cnt = 0.
        for s in range(sensors):
            vec = features.extract_features( data_matrix[s] )

            prep = PREP if d_or_m == PATIENT else PREP_D

            vec = prep_data(vec.reshape((1, vec.shape[0])), prep)

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

            num_items = NUM_IN_HEAP if d_or_m == PATIENT else NUM_IN_HEAP_D
            t = 0.
            for delta, pi, pp in heapq.nlargest(num_items, h):
                t += float(INTERICTAL_CLS if pi > pp else PREICTAL_CLS)
            t /= float(num_items)
            print "# sensor", s, "max N:", heapq.nlargest(num_items+1, h)

            p = INTERICTAL_CLS if t < .5 else PREICTAL_CLS
            t = abs(t - .5)
            heapq.heappush(heap_sensors, (t, p))

        num_items = NUM_IN_HEAP if d_or_m == PATIENT else NUM_IN_HEAP_D
        t = 0.
        for p, cls in heapq.nlargest(num_items, heap_sensors):
            t += cls

        p = t / num_items
        v = INTERICTAL_CLS if p < .5 else PREICTAL_CLS
        result.append(v)
        print "# %s,%d (%f)" % (fn, v, p)

    res = classification_report(y, result)
    print res





def process_rf():

    data_p, data_d = read_data_fast()

    test_ii_indices = [424,295,67,340,107,231,134,443,268,171,86,391,220,267,236,345,388,232,88,57,691,800,840,967,509,582,521,620,488,742,755,740,492,577,637,781,496,787,958,782,1874,981,1197,2401,1914,1798,2312,2348,1950,1090,2034,1512,1875,1604,1867,1441,1298,1772,1842,2061,2990,3008,2728,3188,3160,2507,3070,2513,2836,2520,2430,2677,2999,2943,3028,3104,2977,3165,3209,2530,3626,3616,3427,3275,3396,3229,3497,3649,3424,3547,3365,3496,3652,3512,3537,3267,3542,3369,3262,3611,3719,3675,3676,3712,3721,3682,3683,3684,3687,3688,3717,3691,3723,3693,3711,3695,3696,3698,3702,3706,3725,3726,3727,3731,3734,3737,3739,3740,3741,3742,3743,3745,3746,3747,3765,3751,3753,3764,3759,3763]
    test_pi_indices = [3784,3767,3769,3773,3787,3775,3788,3780,3782,3783,3812,3791,3792,3816,3808,3826,3810,3813,3814,3803,3883,3895,3872,3900,3843,3846,3838,3839,3861,3881,3953,3946,3954,3951,3964,3931,3938,3936,3927,3999,4002,4003,4029,4015,4030,4008,4017,4010,4014,4019,4032,4034,4038,4039,4040,4044,4045,4048,4051,4054,4057,4058,4059,4060,4061,4064]


    ###############################################################3

    LEN = features.X_LEN

    rows_p = data_p.shape[0]
    cols_p = data_p.shape[1]

    rows_d = data_d.shape[0]
    cols_d = data_d.shape[1]


    #cd = RandomForestClassifier(n_estimators=100)
    #cp = RandomForestClassifier(n_estimators=100)

    cd = hmm.GaussianHMM(n_components=N_COMP_D)
    cp = hmm.GaussianHMM(n_components=N_COMP)

##    cd.fit(data_d[:,X_IDX:], data_d[:,CLS_IDX])
##    cp.fit(data_p[:,X_IDX:], data_p[:,CLS_IDX])
    cd.fit(data_d[:,X_IDX:])
    cp.fit(data_p[:,X_IDX:])
    print "# Fitting finished"


    y = []
    result = []

    for i in (test_ii_indices + test_pi_indices):

        if i in files.INTERICTAL_FILES:
            fn = files.INTERICTAL_FILES[ i ]
        else:
            fn = files.PREICTAL_FILES[ i ]

        seg = get_segment(fn)
        cls = get_cls(seg)

        if i in patients:
            d_or_m = PATIENT
        else:
            d_or_m = DOG

        y.append(cls)



        data_matrix, freq, sensors, length = get_data_matrix(fn, seg)

        heap_sensors = []

        for s in range(sensors):
            vec = features.extract_features( data_matrix[s] )

            prep = PREP if d_or_m == PATIENT else PREP_D
            vec = prep_data(vec.reshape((1, vec.shape[0])), prep)


            h = []

            for v in vec:
                if d_or_m == PATIENT:
                    t = cp.predict_proba(v)
                    #t = cp.predict(v)
                    pi = t[0,0]
                    pp = t[0,1]
                else:
                    t = cd.predict_proba(v)
                    #t = cd.predict(v)
                    pi = t[0,0]
                    pp = t[0,1]


                delta = abs(pi - pp)
                #print "#>>", "%2.20f vs %2.20f (%f)" % (pi, pp, delta)
                heapq.heappush(h, (delta, pi, pp))

            num_items = NUM_IN_HEAP if d_or_m == PATIENT else NUM_IN_HEAP_D
            t = 0.
            for delta, pi, pp in heapq.nlargest(num_items, h):
                t += float(INTERICTAL_CLS if pi > pp else PREICTAL_CLS)
            t /= float(num_items)
            print "# sensor", s, "max N:", heapq.nlargest(num_items+1, h)

            p = INTERICTAL_CLS if t < .5 else PREICTAL_CLS
            t = abs(t - .5)
            heapq.heappush(heap_sensors, (t, p))

        num_items = NUM_IN_HEAP if d_or_m == PATIENT else NUM_IN_HEAP_D
        t = 0.
        for p, cls in heapq.nlargest(num_items, heap_sensors):
            t += cls

        p = t / num_items
        v = INTERICTAL_CLS if p < .5 else PREICTAL_CLS
        result.append(v)
        print "# %s,%d (%f)" % (fn, v, p)

    res = classification_report(y, result)
    print res





def main():
    sp.random.seed()

    #gen_train_sets()
    #process_hmm3()
    #process_rf()
    #return

    #data = read_data()

    global PREP, N_COMP, NUM_IN_HEAP
    global PREP_D, N_COMP_D, NUM_IN_HEAP_D


    for num_d in range(1,15):
        for num_p in range(1,15):
            NUM_IN_HEAP = num_p
            NUM_IN_HEAP_D = num_d
            process_hmm3()
            print "# num_p", NUM_IN_HEAP, "num_d", NUM_IN_HEAP_D

    #process()
    #process_hmm()
    #process_hmm2()
    #dog_vs_man()
    #dog_vs_man2()


main()






