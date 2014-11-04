


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


DLL = ctypes.cdll.LoadLibrary("C:\\Temp\\kaggle\\epilepsy\\scripts\\sub5\\dtw_fast.dll")


path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_train = path + "prepared_sub5\\"


interictal_prefix = "interictal_segment"
preictal_prefix = "preictal_segment"


PREICTAL_CLS = 1
INTERICTAL_CLS = 0

X_LEN = features.X_LEN

W=50



PATIENT = 1
DOG = 0





patients = set(files.Patient_1_interictal_files + files.Patient_1_preictal_files + files.Patient_1_test_files +
               files.Patient_2_interictal_files + files.Patient_2_preictal_files + files.Patient_2_test_files)




def get_subj(f_id):
    if f_id in patients:
        return PATIENT
    return DOG


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



def write(fout, cls, vec, f_id, s_id):
    to_file = sp.concatenate(([f_id, s_id, get_subj(f_id), cls], vec))
    to_file.tofile(fout, sep=',')
    fout.write(os.linesep)
    fout.flush()
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

    log = open(path + "..\\logs\\prepare_train.sub5.log", "w+")


    #########################################################
    ## select train files
    #########################################################

    #train_ii_indices = [72,116,203,250,265,328,341,355,436,458,486,526,593,612,613,672,705,795,904,941,1070,1160,1284,1287,1447,1555,1652,2009,2330,2383,2447,2590,2725,2857,2906,2955,2960,3020,3161,3171,3311,3409,3546,3558,3563,3576,3589,3645,3653,3655,3675,3686,3688,3694,3695,3696,3698,3702,3710,3719,3727,3730,3734,3735,3738,3740,3743,3746,3755,3764]
    #train_pi_indices = [3768,3769,3771,3776,3777,3779,3780,3781,3787,3788,3791,3795,3802,3805,3806,3808,3812,3816,3826,3831,3835,3842,3849,3852,3853,3854,3864,3873,3885,3890,3909,3918,3924,3933,3948,3955,3965,3973,3987,3989,4005,4006,4013,4014,4017,4018,4022,4023,4025,4027,4032,4034,4035,4036,4038,4040,4041,4043,4046,4048,4051,4052,4053,4054,4057,4058,4059,4062,4064,4065]
    train_ii_indices = [33,44,51,100,145,165,191,194,200,224,251,265,322,326,344,346,366,373,420,441,519,535,545,583,602,605,626,627,630,651,657,700,770,775,776,786,848,896,905,950,999,1038,1134,1137,1195,1539,1554,1606,1676,1691,1779,1861,1880,1891,1968,2002,2109,2174,2321,2347,2475,2484,2528,2581,2591,2764,2781,2804,2855,2903,2979,2983,3027,3036,3038,3072,3139,3144,3161,3192,3237,3241,3245,3269,3270,3293,3305,3318,3368,3393,3411,3431,3452,3453,3482,3531,3554,3605,3647,3667,3677,3678,3681,3685,3686,3690,3697,3699,3700,3701,3703,3704,3705,3707,3708,3710,3714,3715,3718,3720,3724,3728,3729,3730,3732,3733,3735,3736,3738,3744,3748,3749,3752,3754,3756,3757,3758,3760,3761,3762]
    train_pi_indices = [3768,3770,3771,3772,3776,3778,3779,3781,3786,3789,3795,3800,3801,3805,3806,3809,3815,3823,3824,3829,3841,3848,3855,3864,3869,3871,3880,3885,3894,3903,3905,3915,3917,3926,3928,3930,3972,3977,3995,4000,4001,4005,4012,4018,4021,4022,4023,4024,4027,4028,4031,4033,4035,4036,4037,4041,4042,4043,4046,4047,4049,4050,4052,4053,4055,4056,4062,4063,4065,4066]
    #########################################################

    fn_cnt = 0
    fname_train = "%strain_%d.csv" % (path_train, fn_cnt)
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
                vec = features.extract_features( data_matrix[s] )
                write(fout, INTERICTAL_CLS, vec, i, s)

            if (cnt % 20) == 0:
                dbg(log, "currently processed %d files" % cnt)

                fname_train = "%strain_%d.csv" % (path_train, fn_cnt)
                fn_cnt += 1
                fout.close()
                fout = open(fname_train, "wb+")

        for i in train_pi_indices:
            cnt += 1
            fn = files.PREICTAL_FILES[ i ]
            data_matrix, freq, sensors, length = get_data_matrix(fn, 'preictal')
            for s in range(sensors):
                vec = features.extract_features( data_matrix[s] )
                write(fout, PREICTAL_CLS, vec, i, s)

            if (cnt % 20) == 0:
                dbg(log, "currently processed %d files" % cnt)

                fname_train = "%strain_%d.csv" % (path_train, fn_cnt)
                fn_cnt += 1
                fout.close()
                fout = open(fname_train, "wb+")

    dbg(log, "DONE, %d files" % cnt)


def main():
    #load_means()
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
