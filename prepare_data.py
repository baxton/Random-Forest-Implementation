

import os
import sys
import array
import numpy as np
import scipy as sp
import scipy.io as sio



#subdirs = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']
subdirs = ['Dog_1']

#subdirs = ['Patient_1', 'Patient_2']

path = "C:\\Temp\\kaggle\\epilepsy\\data\\"
path_out = "C:\\Temp\\kaggle\\epilepsy\\data\\prepared\\"

interictal_prefix = "interictal_segment"
preictal_prefix = "preictal_segment"
test_prefix = "test_segment"

fname_train = "dog_train"
fname_cross_test = "dog_cross_test"
fname_test = "dog_test"



PART_FOR_TEST = .3

PREICTAL_CLS = 1
INTERICTAL_CLS = 0



X_LEN = 10000
VEC_LEN = X_LEN + 1



DTW = sp.zeros((VEC_LEN, VEC_LEN ), dtype=np.float)


def DTWDistance(s1, s2,w):

    w = max(w, abs(len(s1)-len(s2)))

    DTW.fill(float('inf'))
    DTW[0,0] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2

            dtwI = i + 1
            dtwJ = j + 1

            DTW[dtwI, dtwJ] = dist + min(DTW[dtwI-1,dtwJ], DTW[dtwI,dtwJ-1], DTW[dtwI-1,dtwJ-1])

    return sp.sqrt(DTW[len(s1), len(s2)])





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



def write(fout, vec, cls, distance_to_1, distance_to_2):
    a = array.array('d')    # unsigned short
    a.append(distance_to_1)
    a.append(distance_to_2)
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





def normalize(vec):
    m = vec.mean()
    max_v = vec.max()
    min_v = vec.min()

    vec -= m
    vec /= (max_v - min_v)














def prepare_vector(ORIG):
    N = ORIG.shape[0]
    bags = 10000
    bag_size = N / bags

    vec = sp.zeros((bags,))

    for i in range(bags):
        vec[i] = ORIG[i:i+bag_size].mean()

    return vec







def start(dir_name):
    with open(path_out + fname_train + "_" + dir_name, "wb+") as fout:
        with open(path_out + fname_cross_test + "_" + dir_name, "wb+") as fout_test:
            with open(path_out + fname_test + "_" + dir_name, "wb+") as fout_sub:
                full_path = path + dir_name + "\\"
                if os.path.exists(full_path):
                    interictal_files, preictal_files, test_files = get_files(full_path)
                    process(interictal_files, preictal_files, test_files, fout, fout_test, fout_sub, dir_name)


def process(interictal_files, preictal_files, sub_files, fout, fout_test, fout_sub, dir_name):

    log = open(path + dir_name + ".log", "w+")

    log.write("Start processing: %s%s" % (dir_name, os.linesep))

    # balance
    LI = len(interictal_files)
    LP = len(preictal_files)

    L = min(LP, LI)
    LEN_TRAIN = int(float(L) * (1. - PART_FOR_TEST))
    LEN_TEST = L - LEN_TRAIN

    log.write("for train: %d; for test: %d%s" % (LEN_TRAIN, LEN_TEST, os.linesep))

    # define indices for train
    interictal_indices = get_k_of_n(L, 0, LI)
    preictal_indices = get_k_of_n(L, 0, LP)

    dot_1 = sp.zeros((X_LEN,))
    dot_2 = sp.zeros((X_LEN,))
    dot_2 += 5000

    #
    VEC_LEN = -1

    # interictal
    for idx in range(L):
        f = interictal_files[interictal_indices[idx]]
        log.write("Processing: %s%s" % (f, os.linesep))
        log.flush()

        data_matrix, freq, sensors, length = get_data_matrix(f, 'interictal')
        for s in range(sensors):
            try:
                a = prepare_vector(np.array(data_matrix[s], dtype=float))
                VEC_LEN = a.shape[0]

                if idx < LEN_TRAIN:
                    write(fout, a, INTERICTAL_CLS, DTWDistance(dot_1, a, 3), DTWDistance(dot_2, a, 3))
                else:
                    write(fout_test, a, INTERICTAL_CLS, DTWDistance(dot_1, a, 3), DTWDistance(dot_2, a, 3))
                a = None
            except Exception as ex:
                log.write("ERR: vector was not prepared for sensor %d%s: %s" % (s, os.linesep, ex.message))
        # end for sensors
    # end interictals

    # preictal
    for idx in range(L):
        f = preictal_files[preictal_indices[idx]]
        log.write("Processing: %s%s" % (f, os.linesep))
        log.flush()

        data_matrix, freq, sensors, length = get_data_matrix(f, 'preictal')

        for s in range(sensors):
            try:
                a = prepare_vector(np.array(data_matrix[s], dtype=float))

                if idx < LEN_TRAIN:
                    write(fout, a, PREICTAL_CLS, DTWDistance(dot_1, a, 3), DTWDistance(dot_2, a, 3))
                else:
                    write(fout_test, a, PREICTAL_CLS, DTWDistance(dot_1, a, 3), DTWDistance(dot_2, a, 3))
                a = None
            except:
                log.write("ERR: vector was not prepared for sensor %d%s: %s" % (s, os.linesep, ex.message))
        # end of for sensors
    # end of preictal

    # test
##    initial_vec = None
##    for f in sub_files:
##        log.write("Processing: %s%s" % (f, os.linesep))
##        log.flush()
##
##        data_matrix, freq, sensors, length = get_data_matrix(f, 'test_segment')
##
##        for s in range(sensors):
##            try:
##                a = prepare_vector(np.array(data_matrix[s], dtype=float))
##
##                write(fout_sub, a, -1., -1.)
##                a = None
##            except:
##                log.write("ERR: vector was not prepared for sensor %d%s: %s" % (s, os.linesep, ex.message))
##        # end of for sensors
##    # end of preictal

    log.write("DONE, vec len %d%s" % (VEC_LEN, os.linesep))




def main():
    #if TEST:
    for subd in subdirs:
        start(subd)
#        break

if __name__ == '__main__':
    main()
