


import os
import sys
import array
import numpy as np
import scipy as sp
import scipy.io as sio
import numpy.fft as fft
import ctypes
import datetime as dt

from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import classification_report

import files
import features



path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_train = path + "prepared_sub7\\"


PREICTAL_CLS = 1
INTERICTAL_CLS = 0


patients = set(files.Patient_1_interictal_files + files.Patient_1_preictal_files + files.Patient_1_test_files +
               files.Patient_2_interictal_files + files.Patient_2_preictal_files + files.Patient_2_test_files)



FILE_IDX  = 0
SEG_IDX   = 1
SUB_IDX   = 2
CLS_IDX   = 3
X_IDX     = 4








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




def get_cor(v1, v2):
    L = min(v1.shape[0], v2.shape[0])
    cr = np.corrcoef(v1[:L], v2[:L])
    return cr[0,1]



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



def get_k_of_n(k, low, high):
    numbers = np.array(range(low, low + k))
    for i in range(low + k, high):
        r = sp.random.randint(low, i) - low
        if r < k:
            numbers[r] = i
    return numbers





def to_features(orig):
    trend, seson, noise = features.extract_features( orig )

    cum_sum = np.cumsum(seson)
    cum_sum /= (cum_sum.max() - cum_sum.min())
    seson_norm = seson - cum_sum

    # stats
    vec = sp.concatenate((
        [
            orig.std(),         # 0 <
            orig.min(),         # 1 <
            orig.max(),         # 2
            np.median(orig),

            trend.std(),        # 4 <
            trend.min(),        # 5 <
            trend.max(),        # 6
            np.median(trend),   # 7 <

            seson.std(),        # 8
            seson.min(),
            seson.max(),
            np.median(seson),   # 11

            noise.std(),
            noise.min(),
            noise.max(),
            np.median(noise),   # 15

            cum_sum.std(),      # 16 <
            cum_sum.min(),
            cum_sum.max(),
            np.median(cum_sum),

            seson_norm.std(),   # 20
            seson_norm.min(),
            seson_norm.max(),
            np.median(seson_norm),

            get_cor(orig, trend),   # 24
            get_cor(orig, seson),
            get_cor(orig, noise),
            get_cor(orig, cum_sum),
            get_cor(orig, seson_norm),  # 28

            get_cor(trend, seson),      # 29
            get_cor(trend, noise),
            get_cor(trend, cum_sum),    # 31 <
            get_cor(trend, seson_norm), # 32

            get_cor(seson, noise),
            get_cor(seson, cum_sum),
            get_cor(seson, seson_norm),

            get_cor(noise, cum_sum),
            get_cor(noise, seson_norm),

            get_cor(cum_sum, seson_norm),


        ],
        ))
    return vec




def save_rf(classifier, res):
    import pickle

    f1 = res.split("\n")[-2].split()[-2]

    fn = path + 'rf7_%s.txt' % f1
    cnt = 0
    while os.path.exists(fn):
        cnt += 1
        fn = path + 'rf7_%s_(%d).txt' % (f1, cnt)

    output = open(fn, 'wb')
    pickle.dump(classifier, output)
    output.close()




def get_train_indices(data):
    # here I want to adjust training set and reduce number of "interictal" vectors randomly
    indices = data[:,CLS_IDX] == INTERICTAL_CLS
    N = indices.sum()
    K = int(N * .1)

    to_be_removed = get_k_of_n(K, 0, N)
    print "# to be removed: "
    for  n in to_be_removed:
        print n,
    print

    for i in range(indices.shape[0]):
        if not indices[i]:
            indices[i] = True
        else:
            if i in to_be_removed:
                indices[i] = False
    return indices


def process():

    data = read_data()
    indices = get_train_indices(data)

    c = RandomForestClassifier(n_estimators=5001, random_state=int(dt.datetime.now().microsecond))

    X = data[indices,X_IDX:]
    Y = data[indices,CLS_IDX]

    c.fit(X, Y)
    print "# Fitting finished"

    print c.feature_importances_


    #########################################################
    ## select test files
    #########################################################

    test_ii_indices = [147,429,451,437,325,122,357,168,291,250,254,103,267,422,339,264,729,786,957,961,484,526,727,603,889,789,879,759,492,597,604,607,518,750,609,771,954,637,849,503,970,646,769,624,923,2105,1171,1076,1815,1427,1316,1513,2340,1200,2251,1854,1984,2306,2082,1583,1692,2131,1433,1455,1131,2168,1038,1100,1572,2189,1963,2228,1430,1652,1858,1379,1272,1620,1124,1249,1597,1780,2260,1549,2002,2027,2144,1656,1409,2187,1505,1496,1273,1233,1531,3144,2421,3006,3148,2994,2629,2528,3217,2428,2951,3122,3010,2569,2968,3090,2750,2605,2595,3195,2778,2822,2924,2849,3179,3084,2819,2872,3125,2666,2800,3066,2873,2700,2815,2456,2693,2610,2919,2971,3215,2588,2699,3109,2466,2647,2756,2469,2534,3132,2554,3047,3011,3111,2999,2727,2560,2516,2830,2619,2980,3032,3158,3024,2752,3005,2602,2771,3670,3518,3539,3661,3245,3313,3426,3256,3415,3470,3469,3407,3565,3589,3386,3561,3411,3274,3650,3566,3442,3674,3679,3684,3687,3691,3692,3695,3700,3703,3705,3707,3717,3719,3720,3723,3724,3726,3730,3738,3739,3741,3746,3749,3751,3754,3757,3758,3764]
    test_pi_indices = [3767,3768,3771,3773,3774,3775,3779,3783,3791,3794,3796,3797,3800,3805,3806,3812,3818,3821,3822,3824,3826,3832,3833,3835,3837,3840,3844,3845,3846,3850,3852,3853,3855,3856,3861,3865,3866,3872,3873,3875,3878,3899,3902,3904,3907,3908,3912,3915,3916,3917,3918,3919,3920,3925,3926,3928,3933,3935,3942,3944,3951,3955,3956,3967,3971,3978,3981,3982,3983,3986,3993,3999,4000,4002,4004,4008,4010,4018,4022,4023,4024,4030]

    #########################################################

    y = []
    result = []

    for i in (test_ii_indices + test_pi_indices):
        #if i in patients:
        #    continue

        if i in files.INTERICTAL_FILES:
            fn = files.INTERICTAL_FILES[ i ]
        else:
            fn = files.PREICTAL_FILES[ i ]
    #for i, fn in files.TEST_FILES.items():

        seg = get_segment(fn)
        cls = get_cls(seg)

        y.append(cls)

        data_matrix, freq, sensors, length = get_data_matrix(fn, seg)


        good_sensors = [4, 6, 14] # get_k_of_n(7, 0, sensors)
        good_sensors = range(30)

        pi_sum = 0.
        pp_sum = 0.
        pi_prod = 1.
        pp_prod = 1.
        cnt = 0.
        for s in range(sensors):
            if s not in good_sensors:
                continue

            v = to_features(data_matrix[s])

            pred_cls = c.predict_proba(v)
            print "# sensor ", s, pred_cls
            pi_sum += pred_cls[0,0]
            pp_sum += pred_cls[0,1]
            pi_prod *= pred_cls[0,0]
            pp_prod *= pred_cls[0,1]
            cnt += 1.

        pi = cnt * pi_prod / pi_sum
        pp = cnt * pp_prod / pp_sum
        p = pi_sum / cnt if pi > pp else pp_sum / cnt
        v = INTERICTAL_CLS if pi > pp else PREICTAL_CLS
        print "#%s,%f (%d) (%f %f)" % (fn, p, v, pi, pp)
        result.append(v)


    #######
    res = classification_report(y, result)
    print res

    #answer = raw_input("Save RF?")
    #if "Y" == answer:
    print "saving RF"
    save_rf(c, res)





def main():
    sp.random.seed(dt.datetime.now().microsecond)
    process()



if __name__ == '__main__':
    main()



#
# cat dog_1_result.txt | grep "mat : " | sed 's/.*\(Dog_1.*mat\) : \([10]\)/\1,\2\r\n/' > dog_1_sub.txt
# cat dog_2_result.txt | grep "mat : " | sed 's/.*\(Dog_2.*mat\) : \([10]\)/\1,\2\r\n/' > dog_2_sub.txt
#
#
# for f in `ls -1 ../data/Dog_3/ | grep test_segment`; do python feed_one.py ../data/Dog_3/$f "test_segment" | ./cls_node.exe >> dog_3_result.txt ; done
#



# (cat 2.txt | grep "inter.*mat" | wc -l;  cat 2.txt | grep "inter.*mat" | grep "(0)" | wc -l) | awk 'BEGIN {I=0} {print $0" to "I; VALS[I] = $0; I=I+1;} END {print VALS[1] / VALS[0]}'
# (cat 1.txt | grep "pre.*mat" | wc -l;  cat 1.txt | grep "pre.*mat" | grep "(1)" | wc -l) | awk 'BEGIN {I=0} {print $0" to "I; VALS[I] = $0; I=I+1;} END {print VALS[1] / VALS[0]}'


