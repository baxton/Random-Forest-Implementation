


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
path_train = path + "prepared_sub5\\"


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

    m = trend[ trend > 0 ].mean()
    pos_t = trend[ trend > (m) ]
    pos_t = features.moving_average(pos_t, 500)
    m = trend[ trend < 0 ].mean()
    neg_t = trend[ trend < (m) ]
    neg_t = features.moving_average(neg_t, 500)

    m = seson[ seson > 0 ].mean()
    pos_s = seson[ seson > (m) ]
    pos_s = features.moving_average(pos_s, 500)
    m = seson[ seson < 0 ].mean()
    neg_s = seson[ seson < (m) ]
    neg_s = features.moving_average(neg_s, 500)


    # stats
    vec = sp.concatenate((
        [
            orig.std(),
            orig.min(),
            orig.max(),
            np.median(orig),
            np.percentile(orig, 25),
            np.percentile(orig, 75),

            trend.std(),
            trend.min(),
            trend.max(),
            np.median(trend),
            np.percentile(trend, 25),
            np.percentile(trend, 75),

            seson.std(),
            seson.min(),
            seson.max(),
            np.median(seson),
            np.percentile(seson, 25),
            np.percentile(seson, 75),

            noise.std(),
            noise.min(),
            noise.max(),
            np.median(noise),
            np.percentile(noise, 25),
            np.percentile(noise, 75),

            cum_sum.std(),
            cum_sum.min(),
            cum_sum.max(),
            np.median(cum_sum),
            np.percentile(cum_sum, 25),
            np.percentile(cum_sum, 75),

            seson_norm.std(),
            seson_norm.min(),
            seson_norm.max(),
            np.median(seson_norm),
            np.percentile(seson_norm, 25),
            np.percentile(seson_norm, 75),

            neg_t.std(),
            neg_t.min(),
            neg_t.max(),
            np.median(neg_t),
            np.percentile(neg_t, 25),
            np.percentile(neg_t, 75),

            pos_t.std(),
            pos_t.min(),
            pos_t.max(),
            np.median(pos_t),
            np.percentile(pos_t, 25),
            np.percentile(pos_t, 75),

            neg_s.std(),
            neg_s.min(),
            neg_s.max(),
            np.median(neg_s),
            np.percentile(neg_s, 25),
            np.percentile(neg_s, 75),

            pos_s.std(),
            pos_s.min(),
            pos_s.max(),
            np.median(pos_s),
            np.percentile(pos_s, 25),
            np.percentile(pos_s, 75),

            get_cor(orig, trend),
            get_cor(orig, seson),
            get_cor(orig, noise),
            get_cor(orig, cum_sum),
            get_cor(orig, seson_norm),
            get_cor(orig, neg_t),
            get_cor(orig, pos_t),
            get_cor(orig, neg_s),
            get_cor(orig, pos_s),

            get_cor(trend, seson),
            get_cor(trend, noise),
            get_cor(trend, cum_sum),
            get_cor(trend, seson_norm),
            get_cor(trend, neg_t),
            get_cor(trend, pos_t),
            get_cor(trend, neg_s),
            get_cor(trend, pos_s),

            get_cor(seson, noise),
            get_cor(seson, cum_sum),
            get_cor(seson, seson_norm),
            get_cor(seson, neg_t),
            get_cor(seson, pos_t),
            get_cor(seson, neg_s),
            get_cor(seson, pos_s),

            get_cor(noise, cum_sum),
            get_cor(noise, seson_norm),
            get_cor(noise, neg_t),
            get_cor(noise, pos_t),
            get_cor(noise, neg_s),
            get_cor(noise, pos_s),

            get_cor(cum_sum, seson_norm),
            get_cor(cum_sum, neg_t),
            get_cor(cum_sum, pos_t),
            get_cor(cum_sum, neg_s),
            get_cor(cum_sum, pos_s),

            get_cor(seson_norm, neg_t),
            get_cor(seson_norm, pos_t),
            get_cor(seson_norm, neg_s),
            get_cor(seson_norm, pos_s),

            get_cor(neg_t, pos_t),
            get_cor(neg_t, neg_s),
            get_cor(neg_t, pos_s),

            get_cor(pos_t, neg_s),
            get_cor(pos_t, pos_s),

            get_cor(neg_s, pos_s),

        ],
        ))
    return vec




def process():

    log = open(path + "..\\logs\\prepare_train.sub5.log", "w+")

    fout_prefix = "fft"


    #########################################################
    ## select train files
    #########################################################

    train_ii_indices = [34,153,161,193,200,223,229,263,300,315,319,444,446,448,459,476,505,516,519,531,534,548,619,676,681,684,698,703,720,726,761,762,764,792,823,852,865,868,877,884,907,936,945,953,964,994,1000,1063,1068,1121,1153,1228,1263,1265,1291,1320,1398,1404,1410,1421,1486,1514,1542,1589,1639,1679,1682,1766,1788,1803,1846,1933,1968,1979,1988,1995,2010,2039,2040,2061,2088,2099,2114,2134,2136,2207,2247,2255,2298,2312,2317,2323,2384,2412,2413,2449,2454,2464,2494,2497,2511,2513,2532,2533,2535,2579,2580,2584,2598,2601,2607,2631,2651,2656,2660,2663,2674,2683,2706,2710,2714,2723,2730,2779,2784,2803,2804,2805,2857,2870,2874,2875,2888,2929,2962,2973,2975,2977,2987,2989,3003,3018,3019,3028,3031,3044,3050,3055,3069,3093,3112,3119,3153,3160,3165,3173,3180,3192,3201,3210,3211,3219,3248,3273,3286,3290,3302,3306,3311,3324,3342,3391,3400,3444,3461,3477,3493,3530,3553,3554,3583,3629,3662,3675,3676,3677,3678,3680,3681,3682,3683,3685,3686,3688,3689,3690,3693,3694,3696,3697,3698,3699,3701,3702,3704,3706,3708,3709,3710,3711,3712,3713,3714,3715,3716,3718,3721,3722,3725,3727,3728,3729,3731,3732,3733,3734,3735,3736,3737,3740,3742,3743,3744,3745,3747,3748,3750,3752,3753,3755,3756,3759,3760,3761,3762,3763,3765]
    train_pi_indices = [3766,3769,3770,3772,3776,3777,3778,3780,3781,3782,3784,3785,3786,3787,3788,3789,3790,3792,3793,3795,3798,3799,3801,3802,3803,3804,3807,3808,3809,3810,3811,3813,3814,3815,3816,3817,3819,3820,3823,3825,3827,3828,3829,3830,3831,3834,3836,3838,3839,3841,3842,3843,3847,3848,3849,3851,3854,3857,3858,3859,3860,3862,3863,3864,3867,3868,3869,3870,3871,3874,3876,3877,3879,3880,3881,3882,3883,3884,3885,3886,3887,3888,3889,3890,3891,3892,3893,3894,3895,3896,3897,3898,3900,3901,3903,3905,3906,3909,3910,3911,3913,3914,3921,3922,3923,3924,3927,3929,3930,3931,3932,3934,3936,3937,3938,3939,3940,3941,3943,3945,3946,3947,3948,3949,3950,3952,3953,3954,3957,3958,3959,3960,3961,3962,3963,3964,3965,3966,3968,3969,3970,3972,3973,3974,3975,3976,3977,3979,3980,3984,3985,3987,3988,3989,3990,3991,3992,3994,3995,3996,3997,3998,4001,4003,4005,4006,4007,4009,4011,4012,4013,4014,4015,4016,4017,4019,4020,4021,4025,4026,4027,4028,4029,4031,4032,4033,4034,4035,4036,4037,4038,4039,4040,4041,4042,4043,4044,4045,4046,4047,4048,4049,4050,4051,4052,4053,4054,4055,4056,4057,4058,4059,4060,4061,4062,4063,4064,4065,4066]
    test_ii_indices = [147,429,451,437,325,122,357,168,291,250,254,103,267,422,339,264,729,786,957,961,484,526,727,603,889,789,879,759,492,597,604,607,518,750,609,771,954,637,849,503,970,646,769,624,923,2105,1171,1076,1815,1427,1316,1513,2340,1200,2251,1854,1984,2306,2082,1583,1692,2131,1433,1455,1131,2168,1038,1100,1572,2189,1963,2228,1430,1652,1858,1379,1272,1620,1124,1249,1597,1780,2260,1549,2002,2027,2144,1656,1409,2187,1505,1496,1273,1233,1531,3144,2421,3006,3148,2994,2629,2528,3217,2428,2951,3122,3010,2569,2968,3090,2750,2605,2595,3195,2778,2822,2924,2849,3179,3084,2819,2872,3125,2666,2800,3066,2873,2700,2815,2456,2693,2610,2919,2971,3215,2588,2699,3109,2466,2647,2756,2469,2534,3132,2554,3047,3011,3111,2999,2727,2560,2516,2830,2619,2980,3032,3158,3024,2752,3005,2602,2771,3670,3518,3539,3661,3245,3313,3426,3256,3415,3470,3469,3407,3565,3589,3386,3561,3411,3274,3650,3566,3442,3674,3679,3684,3687,3691,3692,3695,3700,3703,3705,3707,3717,3719,3720,3723,3724,3726,3730,3738,3739,3741,3746,3749,3751,3754,3757,3758,3764]
    test_pi_indices = [3767,3768,3771,3773,3774,3775,3779,3783,3791,3794,3796,3797,3800,3805,3806,3812,3818,3821,3822,3824,3826,3832,3833,3835,3837,3840,3844,3845,3846,3850,3852,3853,3855,3856,3861,3865,3866,3872,3873,3875,3878,3899,3902,3904,3907,3908,3912,3915,3916,3917,3918,3919,3920,3925,3926,3928,3933,3935,3942,3944,3951,3955,3956,3967,3971,3978,3981,3982,3983,3986,3993,3999,4000,4002,4004,4008,4010,4018,4022,4023,4024,4030]

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
            vec = to_features(data_matrix[sen])
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
