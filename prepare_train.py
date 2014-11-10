


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


#DLL = ctypes.cdll.LoadLibrary("C:\\Temp\\kaggle\\epilepsy\\scripts\\sub6\\dtw_fast.dll")


path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_train = path + "prepared_sub6\\"


interictal_prefix = "interictal_segment"
preictal_prefix = "preictal_segment"


PREICTAL_CLS = 1
INTERICTAL_CLS = 0

X_LEN = features.X_LEN




PATIENT = 1
DOG = 0





patients = set(files.Patient_1_interictal_files + files.Patient_1_preictal_files + files.Patient_1_test_files +
               files.Patient_2_interictal_files + files.Patient_2_preictal_files + files.Patient_2_test_files)




def get_subj(f_id):
    if f_id in patients:
        return PATIENT
    return DOG


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



def DTWDistance(s1, s2, w, giveup_threshold = float('inf')):
    res = ctypes.c_float(0.)
    #code = DLL.DTWDistance(s1.ctypes.data, s2.ctypes.data, ctypes.c_int(w), ctypes.c_float(giveup_threshold), ctypes.addressof(res))
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






def get_cor(v1, v2):
    L = min(v1.shape[0], v2.shape[0])
    cr = np.corrcoef(v1[:L], v2[:L])
    return cr[0,1]




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



def to_features2(orig):
    trend, seson, noise = features.extract_features( orig )

    v1 = fft.fft(trend).astype(np.float32)
    v2 = fft.fft(seson).astype(np.float32)
    v3 = fft.fft(noise).astype(np.float32)

    return sp.concatenate((v1[:10], v2[:10], v3[:10]))


def process():

    log = open(path + "..\\logs\\prepare_train.sub6.log", "w+")

    fout_prefix = "fft"


    #########################################################
    ## select train files
    #########################################################

    train_ii_indices = [6,65,194,271,275,343,412,444,451,469,498,547,592,661,663,825,851,866,900,907,1158,1232,1438,1464,1686,1746,1834,2049,2099,2250,2475,2509,2588,2618,2628,2792,2897,2903,3042,3175,3315,3383,3421,3437,3458,3544,3599,3645,3652,3655,3674,3688,3689,3693,3695,3702,3705,3712,3716,3720,3730,3735,3736,3744,3748,3749,3751,3756,3758,3762]
    train_pi_indices = [3767,3768,3769,3771,3772,3779,3780,3782,3785,3789,3796,3797,3800,3804,3809,3811,3818,3822,3828,3829,3851,3852,3854,3872,3874,3877,3884,3886,3890,3899,3906,3916,3930,3938,3942,3946,3957,3984,3997,3998,4004,4010,4012,4013,4019,4020,4025,4027,4028,4030,4031,4032,4036,4039,4040,4042,4044,4045,4047,4048,4049,4050,4051,4052,4054,4055,4060,4061,4064,4065]

    test_ii_indices = [323,415,296,223,108,154,206,43,456,134,841,816,849,542,505,533,703,487,511,795,1061,1921,1704,1385,1504,1253,1286,2073,1505,1466,2781,2870,2866,2835,2913,2624,2671,2807,3082,3074,3647,3640,3427,3608,3525,3442,3620,3459,3504,3665,3675,3676,3719,3678,3679,3699,3709,3708,3710,3697,3764,3747,3752,3727,3750,3757,3737,3743,3733,3734]
    test_pi_indices = [3766,3770,3773,3774,3775,3784,3777,3778,3781,3787,3790,3806,3803,3826,3805,3831,3821,3799,3820,3812,3902,3857,3863,3864,3882,3893,3869,3850,3840,3861,3924,3987,3927,3908,3981,3922,3975,3962,3917,3945,4001,4002,4016,4023,4021,4007,4017,4024,4011,4015,4033,4034,4035,4037,4038,4041,4043,4046,4053,4056,4057,4058,4059,4062,4063,4066]

    #########################################################

    fn_cnt = 0
    fn_out = "%s%s_%d.csv" % (path_train, fout_prefix, fn_cnt)
    fout = open(fn_out, "wb+")
    fn_cnt += 1

    cnt = 0

    for i in (train_ii_indices + train_pi_indices):
        cnt += 1
        if i in files.INTERICTAL_FILES:
            fn = files.INTERICTAL_FILES[ i ]
        else:
            fn = files.PREICTAL_FILES[ i ]

        seg = get_segment(fn)
        cls = get_cls(seg)

        data_matrix, freq, sensors, length = get_data_matrix(fn, seg)

        for sen in range(sensors):
            vec = to_features2(data_matrix[sen])
            write(fout, cls, vec, i, sen)

        if (cnt % 20) == 0:
            dbg(log, "currently processed %d files" % cnt)
            fn_out = "%s%s_%d.csv" % (path_train, fout_prefix, fn_cnt)
            fn_cnt += 1
            fout.close()
            fout = open(fn_out, "wb+")

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
