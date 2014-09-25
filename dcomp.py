


import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib.pyplot as plt




def plot(ORIG, T, S, R, fname):
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(40, 6)

    a = fig.add_subplot(4,1,1)
    X = range(ORIG.shape[0])
    a.plot(X, ORIG, label="first")

    a = fig.add_subplot(4,1,2)
    X = range(T.shape[0])
    a.plot(X, T, label="second")

    a = fig.add_subplot(4,1,3)
    X = range(S.shape[0])
    a.plot(X, S)

    a = fig.add_subplot(4,1,4)
    X = range(R.shape[0])
    a.plot(X, R)

    plt.savefig(fname)

    #plt.show()




def decompose(vec):
    L = vec.shape[0]
    print "Length: ", L

    T = sp.zeros((L,))
    S = sp.zeros((L,))
    R = sp.zeros((L,))
    tmp = sp.zeros((L,))

    m = int(L / 2)
    for i in range(L):
        b = i-m if i-m >= 0 else 0
        e = i+m if i+m < L else L

        T[i] = vec[b:e].mean()
        tmp[i] = vec[i] - T[i]

    m = int(399 * 1.)
    for i in range(L):
        b = i-m if i-m >= 0 else 0
        e = i+m if i+m < L else L

        S[i] = tmp[b:e].mean()
        R[i] = vec[i] - T[i] - S[i]


    return T, S, R




def get_data_matrix(f, key_word):
    mat = sio.loadmat(f)
    key = [k for k in mat.keys() if key_word in k][0]
    data = mat[key]

    freq = data['sampling_frequency'][0][0][0][0]
    sensors = data['data'][0,0].shape[0]
    length = data['data'][0,0].shape[1]
    length_sec = data['data_length_sec'][0][0][0][0]

    return np.array(data['data'][0,0], dtype=float), freq, sensors, length





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



def decompose2(vec):
    L = vec.shape[0]

    m = int(L / 10)
    T = get_trand(vec, m)

    tmp = vec[m-1:] - T

    m = int(399 * 1.)
    S = get_trand(tmp, m)

    R = tmp[m-1:] - S

    return T, S, R



def get_sin(vec):
    L = vec.shape[0]
    m = vec.mean()
    max_v = vec.max()
    min_v = vec.min()

    tmp = (vec - m) / (max_v - min_v)

    positives = []

    start_pos = -1

    pos = False

    amplitude = 0.
    count = 0.

    for i in range(L):
        if pos:
            if vec[i] > 0:
                if start_pos == -1:
                    start_pos = i

                amplitude += vec[i]
                count += 1.

            else:
                positives[-1] = i - positives[-1]
                pos = False

        else:
            if vec[i] > 0:
                if start_pos == -1:
                    start_pos = i
                positives.append(i)
                amplitude = vec[i]
                count += 1.
                pos = True

    freq = np.mean(positives[:-1]) * 2.
    amplitude /= count

    tmp = np.array(range(L))
    tmp *= sp.pi/freq/2.5
    #result = amplitude * sp.sin( tmp ) + start_pos
    result = sp.sin( tmp )


    m = result.mean()
    max_v = result.max()
    min_v = result.min()

    return (result - m) / (max_v - min_v)




def main():
    path_out = "C:\\Temp\\kaggle\\epilepsy\\data\\"
    fname_out1 = path_out + "Dog_1_interictal_segment_0002_"
    fname_out2 = path_out + "Dog_1_preictal_segment_0002_"

    fname_out = "dog_1_3_"

    fname1 = 'C:\\Temp\\kaggle\\epilepsy\\data\\Dog_1\\Dog_1_interictal_segment_0003.mat'
    fname2 = 'C:\\Temp\\kaggle\\epilepsy\\data\\Dog_1\\Dog_1_preictal_segment_0003.mat'
    key_word1 = 'interictal'
    key_word2 = 'preictal'

    data_matrix1, freq1, sensors1, length1 = get_data_matrix(fname1, key_word1)
    data_matrix2, freq2, sensors2, length2 = get_data_matrix(fname2, key_word2)

    for s in range(16):
        ORIG1 = data_matrix1[s]
        ORIG2 = data_matrix2[s]

        ORIG1 = ORIG1[1:] - ORIG1[0:-1]
        ORIG2 = ORIG2[1:] - ORIG2[0:-1]

        T, S, R = decompose2(ORIG1)
        #T, S, R = decompose2(T)
        iT, iS, iR = decompose2(T)
        #plot(ORIG1, T, S, get_sin(T), fname_out1)

        T, S, R = decompose2(ORIG2)
        #T, S, R = decompose2(T)
        pT, pS, pR = decompose2(T)

        fn = "%s%ss%d.jpg" % (path_out, fname_out, s)
        print fn
        plot(iT, get_sin(iT), get_sin(pT), pT, fn)


if __name__ == '__main__':
    main()
