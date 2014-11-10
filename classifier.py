


import os
import sys
import array
import numpy as np
import scipy as sp
import scipy.io as sio
import ctypes
import datetime as dt

from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import classification_report

import files
import features



path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_train = path + "prepared_sub6\\"


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
            orig.std(),
            orig.min(),
            orig.max(),
            np.median(orig),

            trend.std(),
            trend.min(),
            trend.max(),
            np.median(trend),

            seson.std(),
            seson.min(),
            seson.max(),
            np.median(seson),

            noise.std(),
            noise.min(),
            noise.max(),
            np.median(noise),

            cum_sum.std(),
            cum_sum.min(),
            cum_sum.max(),
            np.median(cum_sum),

            seson_norm.std(),
            seson_norm.min(),
            seson_norm.max(),
            np.median(seson_norm),

            get_cor(orig, trend),
            get_cor(orig, seson),
            get_cor(orig, noise),
            get_cor(orig, cum_sum),
            get_cor(orig, seson_norm),

            get_cor(trend, seson),
            get_cor(trend, noise),
            get_cor(trend, cum_sum),
            get_cor(trend, seson_norm),

            get_cor(seson, noise),
            get_cor(seson, cum_sum),
            get_cor(seson, seson_norm),

            get_cor(noise, cum_sum),
            get_cor(noise, seson_norm),

            get_cor(cum_sum, seson_norm),
        ],
        ))
    return vec




def save_rf(classifier):
    import pickle
    output = open(path + 'rf.txt', 'wb')
    pickle.dump(classifier, output)
    output.close()



def process():

    data = read_data()

    c = RandomForestClassifier(n_estimators=1001)

    train = data[:,X_IDX:]
    #train = sp.concatenate((data[:,X_IDX:], data[:,X_IDX:]**2), axis=1)
    Y = data[:,CLS_IDX]

    c.fit(train, Y)
    print "# Fitting finished"


    #########################################################
    ## select test files
    #########################################################

    test_ii_indices = [323,415,296,223,108,154,206,43,456,134,841,816,849,542,505,533,703,487,511,795,1061,1921,1704,1385,1504,1253,1286,2073,1505,1466,2781,2870,2866,2835,2913,2624,2671,2807,3082,3074,3647,3640,3427,3608,3525,3442,3620,3459,3504,3665,3675,3676,3719,3678,3679,3699,3709,3708,3710,3697,3764,3747,3752,3727,3750,3757,3737,3743,3733,3734]
    test_pi_indices = [3766,3770,3773,3774,3775,3784,3777,3778,3781,3787,3790,3806,3803,3826,3805,3831,3821,3799,3820,3812,3902,3857,3863,3864,3882,3893,3869,3850,3840,3861,3924,3987,3927,3908,3981,3922,3975,3962,3917,3945,4001,4002,4016,4023,4021,4007,4017,4024,4011,4015,4033,4034,4035,4037,4038,4041,4043,4046,4053,4056,4057,4058,4059,4062,4063,4066]

    #########################################################

    y = []
    result = []

    for i in (test_ii_indices + test_pi_indices):
        if i in patients:
            continue

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
        #good_sensors = range(30)

        p = 0.
        cnt = 0.
        for s in range(sensors):
            if s not in good_sensors:
                continue

            v = to_features(data_matrix[s])

            pred_cls = c.predict(v)
            print "# sensor ", s, pred_cls
            p += pred_cls
            cnt += 1.

        p /= cnt
        v = INTERICTAL_CLS if p < .5 else PREICTAL_CLS
        print "#%s,%f (%d)" % (fn, p, v)
        result.append(v)



    #######
    res = classification_report(y, result)
    print res

    answer = raw_input("Save RF?")
    if "Y" == answer:
        save_rf(c)





def main():
    sp.random.seed()
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
