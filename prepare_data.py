

import os
import array
import numpy as np
import scipy as sp
import scipy.io as sio


subdirs = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']

path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_out = "C:\\Temp\\kaggle\\epilepsy\\data\\"

interictal_prefix = "interictal_segment"
preictal_prefix = "preictal_segment"
test_prefix = "test_segment"

fname_train = "dog_train_2.bin"
fname_cross_test = "dog_cross_test_2.bin"


STEP = 1000
FRAME_LEN_SEC = 600
FRAME_FROM_SEQ = 1

PART_FOR_TEST = .3

PREICTAL_CLS = 1
INTERICTAL_CLS = 0


def get_files(path):
    interictal_files = [path + f for f in os.listdir(path) if interictal_prefix in f]
    preictal_files = [path + f for f in os.listdir(path) if preictal_prefix in f]
    test_files = [path + f for f in os.listdir(path) if test_prefix in f]

    return interictal_files, preictal_files, test_files



def get_k_of_n(k, low, high):
    numbers = np.array(range(low, low + k))
    for i in range(low + k, high):
        r = sp.random.randint(low, i) - low
        if r < k:
            numbers[r] = i
    return numbers



def write_txt(fout, vec, cls):
    fout.write("%d," % cls)
    for n in vec:
        fout.write("%d," % n)
    fout.write("\r\n")


def write(fout, vec, cls):
    a = array.array('d')    # unsigned short
    a.append(cls)
    a.extend(vec)
    a.tofile(fout)
    fout.flush()





def get_data_matrix(f, key_word):
    mat = sio.loadmat(f)
    key = [k for k in mat.keys() if key_word in k][0]
    data = mat[key]

    freq = data['sampling_frequency'][0][0][0][0]
    sensors = data['data'][0,0].shape[0]
    length = data['data'][0,0].shape[1]
    length_sec = data['data_length_sec'][0][0][0][0]

    return data['data'][0,0], freq, sensors, length



def get_trand(vec, m):
    fm = float(m)
    wsum = vec[:m].sum()
    num = vec.shape[0] - m + 1

    result = sp.zeros((num, ), dtype=float)
    result[0] = wsum / fm

    for i in range(1, num):
        wsum -= vec[i - 1]
        wsum += vec[i + m - 1]
        result[i] = wsum / fm

    return result



def decompose(vec):
    L = vec.shape[0]

    m = int(L / 10)
    T = get_trand(vec, m)

    tmp = vec[m-1:] - T

    m = int(399 * 1.)
    S = get_trand(tmp, m)

    R = tmp[m-1:] - S

    return T, S, R


def prepare_vector(ORIG):
    ORIG = ORIG[1:] - ORIG[0:-1]

    T, S, R = decompose(ORIG)
    #T, S, R = decompose(S)

    L = S.shape[0]

    chunks = 10
    chunk_len = L / chunks
    result = sp.zeros((chunks,), dtype=float)

    for c in range(10):
        s = c * chunk_len
        e = s + chunk_len
        result[c] = S[s:e].mean()

    return result


# V2 - full vector of positive parts with averaging every 2K nums
def process_train_data_v2(interictal_files, preictal_files, fout, fout_test):

    length = int(len(interictal_files) * (1. - PART_FOR_TEST))
    interictal_for_train_indices = get_k_of_n(length, 0, len(interictal_files))

    length = int(len(preictal_files) * (1. - PART_FOR_TEST))
    preictal_for_train_indices = get_k_of_n(length, 0, len(preictal_files))

    VEC_LEN = 0;

    idx = 0
    for f in interictal_files:
        print "Processing: ", f, "for train" if idx in interictal_for_train_indices else "for test"

        data_matrix, freq, sensors, length = get_data_matrix(f, 'interictal')

        for s in range(sensors):
            a = prepare_vector(np.array(data_matrix[s], dtype=int))

            if VEC_LEN == 0:
                VEC_LEN = a.shape[0]
                print VEC_LEN
            else:
                if VEC_LEN != a.shape[0]:
                    print "ERROR: vec length %d vs %d" % (VEC_LEN, a.shape[0])
                    a = a[:VEC_LEN]

            if idx in interictal_for_train_indices:
                write(fout, a, INTERICTAL_CLS)
            else:
                write(fout_test, a, INTERICTAL_CLS)
        idx += 1
    # end interictals


    idx = 0
    for f in preictal_files:
        print "Processing: ", f, "for train" if idx in preictal_for_train_indices else "for test"

        data_matrix, freq, sensors, length = get_data_matrix(f, 'preictal')

        for s in range(sensors):
            a = prepare_vector(np.array(data_matrix[s], dtype=int))

            if VEC_LEN == 0:
                VEC_LEN = a.shape[0]
            else:
                if VEC_LEN != a.shape[0]:
                    print "ERROR: vec length %d vs %d" % (VEC_LEN, a.shape[0])
                    if VEC_LEN < a.shape[0]:
                        a = a[:VEC_LEN]
                    else:
                        tmp = sp.zeros((VEC_LEN))
                        tmp[:a.shape[0]] = a
                        a = tmp

            if idx in interictal_for_train_indices:
                write(fout, a, PREICTAL_CLS)
            else:
                write(fout_test, a, PREICTAL_CLS)
        idx += 1
    # end of preictal

    print "DONE", "Vec length: %d" % VEC_LEN




def main():

    with open(path_out + fname_train, "wb+") as fout:
        with open(path_out + fname_cross_test, "wb+") as fout_test:
            for subd in subdirs:
                full_path = path + subd + "\\"
                if os.path.exists(full_path):
                    interictal_files, preictal_files, test_files = get_files(full_path)
                    #print '\n'.join(preictal_files)
                    #
                    process_train_data_v2(interictal_files, preictal_files, fout, fout_test)


if __name__ == '__main__':
    main()
