
import os
import sys
import numpy as np
import scipy as sp
import scipy.spatial.distance as spd
import numpy.ctypeslib as npct
import array
import scipy.io as sio
from sklearn.metrics import classification_report
import ctypes

import files
import features


DLL = ctypes.cdll.LoadLibrary("C:\\Temp\\kaggle\\epilepsy\\scripts\\sub1\\dtw_fast.dll")

X_LEN = features.X_LEN
VEC_LEN = features.READ_LEN

PREICTAL_CLS = 1
INTERICTAL_CLS = 0

W = 500

dot_1 = sp.zeros((X_LEN,), dtype=np.float32)
dot_2 = sp.zeros((X_LEN,), dtype=np.float32)
dot_2 += 5000.



test_ii_indices = [69,287,459,369,234,457,414,297,235,415,825,703,691,801,820,673,608,624,552,798,1725,1699,1063,2161,2021,2318,2419,2014,2399,1682,3171,2678,3136,2436,3193,2947,2568,2602,2876,2983,3581,3597,3240,3325,3469,3249,3454,3378,3311,3649,3708,3688,3705,3712,3699,3716,3690,3722,3686,3697,3747,3765,3726,3742,3728,3751,3737,3736,3733,3756]
test_pi_indices = [3789,3768,3780,3770,3772,3774,3775,3776,3787,3779,3811,3824,3827,3810,3829,3795,3796,3816,3820,3831,3866,3870,3846,3868,3873,3854,3894,3869,3885,3875,3904,3961,3920,3983,3926,3916,3922,3959,3929,3953,4002,4021,4022,4006,4019,4009,4011,4027,4015,4018,4031,4033,4035,4038,4041,4044,4046,4047,4049,4051,4055,4056,4058,4060,4062,4065]



def DTWDistance(s1, s2, w, giveup_threshold=float('inf')):
    res = ctypes.c_float(0.)
    DLL.DTWDistance(s1.ctypes.data, s2.ctypes.data, ctypes.c_int(w), ctypes.c_float(giveup_threshold), ctypes.addressof(res))
    return res.value





def get_data_matrix(f, key_word):
    mat = sio.loadmat(f)
    key = [k for k in mat.keys() if key_word in k][0]
    data = mat[key]

    freq = data['sampling_frequency'][0][0][0][0]
    sensors = data['data'][0,0].shape[0]
    length = data['data'][0,0].shape[1]
    length_sec = data['data_length_sec'][0][0][0][0]

    return data['data'][0,0], freq, sensors, length








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




def main():
    sp.random.seed()

    #fname = files.INTERICTAL_FILES[ test_ii_indices[0] ]
    fname = files.PREICTAL_FILES[ test_pi_indices[12] ]
    segment = get_segment(fname)

    print "#", fname, segment

    cls = get_cls(segment)

    #data_matrix, freq, sensors, length = get_data_matrix(f, 'test_segment')
    data_matrix, freq, sensors, length = get_data_matrix(fname, segment)

    #for s in range(sensors):
    for s in sp.random.randint(0, sensors, 3):
    #if True:
    #    s = 1
        a = features.extract_features(data_matrix[s])

        dist1 = DTWDistance(dot_1, a, W)
        dist2 = DTWDistance(dot_2, a, W)

        print "##", dist1, dist2

        vec = sp.concatenate(([cls, dist1, dist2], a)).astype(np.float32)

        ss = []
        for j in range(VEC_LEN):
            ss.append(str(vec[j]))
        print "+" + fname
        print ','.join(ss)





#
# python feed_one.py | ./cls_node.exe
#





if __name__ == '__main__':
    main()



