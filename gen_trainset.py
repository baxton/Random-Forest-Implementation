
import numpy as np
import scipy as sp
import files



NUM_FILES = 10


def get_k_of_n(k, low, high):
    numbers = np.array(range(low, low + k))
    for i in range(low + k, high):
        r = sp.random.randint(low, i) - low
        if r < k:
            numbers[r] = i
    return numbers




def get_ids(train, test, files_ids, num_files):
    tmp = []
    L = len(files_ids)
    indices = get_k_of_n(num_files, 0, L)
    for i in range(L):
        if i in indices:
            train.append(files_ids[i])
        else:
            tmp.append(files_ids[i])
    if num_files <= len(tmp):
        indices = get_k_of_n(num_files, 0, len(tmp))
        test.extend([ tmp[i] for i in indices ])
    else:
        test.extend(tmp)





def main():
    train_ii = []
    train_pi = []

    test_ii = []
    test_pi = []

    # dog 1
    get_ids(train_ii, test_ii, files.Dog_1_interictal_files, NUM_FILES + NUM_FILES)
    get_ids(train_pi, test_pi, files.Dog_1_preictal_files, NUM_FILES)

    get_ids(train_ii, test_ii, files.Dog_2_interictal_files, NUM_FILES + NUM_FILES)
    get_ids(train_pi, test_pi, files.Dog_2_preictal_files, NUM_FILES)

    get_ids(train_ii, test_ii, files.Dog_3_interictal_files, NUM_FILES + NUM_FILES)
    get_ids(train_pi, test_pi, files.Dog_3_preictal_files, NUM_FILES)

    get_ids(train_ii, test_ii, files.Dog_4_interictal_files, NUM_FILES + NUM_FILES)
    get_ids(train_pi, test_pi, files.Dog_4_preictal_files, NUM_FILES)

    get_ids(train_ii, test_ii, files.Dog_5_interictal_files, NUM_FILES + NUM_FILES)
    get_ids(train_pi, test_pi, files.Dog_5_preictal_files, NUM_FILES)

    get_ids(train_ii, test_ii, files.Patient_1_interictal_files, NUM_FILES + NUM_FILES)
    get_ids(train_pi, test_pi, files.Patient_1_preictal_files, NUM_FILES)

    get_ids(train_ii, test_ii, files.Patient_2_interictal_files, NUM_FILES + NUM_FILES)
    get_ids(train_pi, test_pi, files.Patient_2_preictal_files, NUM_FILES)

    ss = [str(t) for t in train_ii]
    print "train_ii_indices = [%s]" % (','.join(ss))
    ss = [str(t) for t in train_pi]
    print "train_pi_indices = [%s]" % (','.join(ss))

    ss = [str(t) for t in test_ii]
    print "test_ii_indices = [%s]" % (','.join(ss))
    ss = [str(t) for t in test_pi]
    print "test_pi_indices = [%s]" % (','.join(ss))


if __name__ == '__main__':
    main()
