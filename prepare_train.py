


import os
import sys
import array
import numpy as np
import numpy.fft as fft
import scipy as sp
import scipy.io as sio
import ctypes

import files
import features


DLL = ctypes.cdll.LoadLibrary("C:\\Temp\\kaggle\\epilepsy\\scripts\\sub1\\dtw_fast.dll")


path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_train = path + "prepared_sub1\\"


interictal_prefix = "interictal_segment"
preictal_prefix = "preictal_segment"


PREICTAL_CLS = 1
INTERICTAL_CLS = 0

X_LEN = features.X_LEN

dot_1 = sp.zeros((X_LEN,), dtype=np.float32)
dot_2 = sp.zeros((X_LEN,), dtype=np.float32)
dot_2 += 5000.

W=500





def DTWDistance(s1, s2, w, giveup_threshold = float('inf')):
    res = ctypes.c_float(0.)
    code = DLL.DTWDistance(s1.ctypes.data, s2.ctypes.data, ctypes.c_int(w), ctypes.c_float(giveup_threshold), ctypes.addressof(res))
    return res.value, code




def dbg(log_out, msg):
    log_out.write("%s%s" % (msg, os.linesep))
    log_out.flush()




def get_k_of_n(k, low, high):
    numbers = np.array(range(low, low + k))
    for i in range(low + k, high):
        r = sp.random.randint(low, i) - low
        if r < k:
            numbers[r] = i
    return numbers



def write(fout, cls, features, dist1, dist2):
    a = array.array('d')    # double
    a.append(cls)
    a.append(dist1)
    a.append(dist2)
    a.extend(features)
    a.tofile(fout)
    fout.flush()

    a = None
# END of write




def get_data_matrix(f, key_word):
    mat = sio.loadmat(f)
    key = [k for k in mat.keys() if key_word in k][0]
    data = mat[key]

    freq = data['sampling_frequency'][0][0][0][0]
    sensors = data['data'][0,0].shape[0]
    length = data['data'][0,0].shape[1]
    length_sec = data['data_length_sec'][0][0][0][0]

    return data['data'][0,0], freq, sensors, length
# END of get_data_matrix








def process():

    log = open(path + "..\\logs\\prepare_train.sub1.log", "w+")


    #########################################################
    ## select train files
    #########################################################

    train_ii_indices = [36,48,79,105,161,231,239,371,393,472,502,520,521,532,538,564,579,655,873,925,1107,1406,1606,1613,1618,2073,2142,2150,2307,2414,2511,2526,2661,2728,2752,2923,2970,3072,3107,3191,3302,3340,3367,3384,3524,3610,3611,3613,3647,3653,3676,3679,3683,3684,3701,3710,3713,3714,3719,3720,3729,3734,3739,3740,3743,3745,3748,3753,3757,3758]
    train_pi_indices = [3766,3771,3773,3778,3782,3783,3784,3785,3786,3788,3799,3801,3803,3805,3806,3807,3813,3815,3818,3822,3833,3836,3844,3849,3864,3877,3884,3895,3899,3903,3907,3912,3917,3921,3958,3964,3969,3982,3994,4000,4001,4005,4007,4010,4013,4014,4016,4017,4025,4030,4032,4034,4036,4037,4039,4040,4042,4043,4045,4048,4050,4052,4053,4054,4057,4059,4061,4063,4064,4066]

    #########################################################

    fn_cnt = 0
    fname_train = "%sDV_train_%d.b " % (path_train, fn_cnt)
    fn_cnt += 1

    cnt = 0

    #with open(fname_train, "wb+") as fout:
    if True:
        fout = open(fname_train, "wb+")

        for i in train_ii_indices:
            cnt += 1
            fn = files.INTERICTAL_FILES[ i ]
            data_matrix, freq, sensors, length = get_data_matrix(fn, 'interictal')
            for s in range(sensors):
                #vec = features.extract_features(data_matrix[s])
                vec = features.DV(data_matrix[s], X_LEN)
                dist1, code = DTWDistance(dot_1, vec, W)
                dist2, code = DTWDistance(dot_2, vec, W)
                write(fout, INTERICTAL_CLS, vec, dist1, dist2)

            if (cnt % 10) == 0:
                dbg(log, "currently processed %d files" % cnt)

                fname_train = "%sDV_train_%d.b " % (path_train, fn_cnt)
                fn_cnt += 1
                fout.close()
                fout = open(fname_train, "wb+")

        for i in train_pi_indices:
            cnt += 1
            fn = files.PREICTAL_FILES[ i ]
            data_matrix, freq, sensors, length = get_data_matrix(fn, 'preictal')
            for s in range(sensors):
                #vec = features.extract_features(data_matrix[s])
                vec = features.DV(data_matrix[s], X_LEN)
                dist1, code = DTWDistance(dot_1, vec, W)
                dist2, code = DTWDistance(dot_2, vec, W)
                write(fout, PREICTAL_CLS, vec, dist1, dist2)

            if (cnt % 10) == 0:
                dbg(log, "currently processed %d files" % cnt)

                fname_train = "%sDV_train_%d.b " % (path_train, fn_cnt)
                fn_cnt += 1
                fout.close()
                fout = open(fname_train, "wb+")

    dbg(log, "DONE, %d files" % cnt)


def main():
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
