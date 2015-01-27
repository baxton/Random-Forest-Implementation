

import numpy as np
import scipy as sp
from array import array


#theta = [-9.463758813487189,    6.18289498247735,  -5.981727709166062,   2.814258387319398,   6.636833969935607,  -6.345538225865881,   3.073328908993378,   6.501391125940127,  -6.312328758737419,   3.078261257398568,]

# E: 0.430294186454 F=14 N=100
#theta = [-7.509859171043416,   4.854661061036129,  -5.764668064537342,   3.516276318780315,   5.270453463003578,  -6.123008178028577,   3.797506564083666,   5.153244029370865,  -6.094239021797968,   3.790274381973932,]

# E: 0.425859415481 F=14 N=100
#theta = [-7.044536404890192,    4.50159870449523,  -5.585559553927324,    3.81326155314644,   4.900691028023418,  -5.947087208517458,   4.097143423809682,   4.789609836178913,  -5.919332980811895,   4.086775213754777,]

theta = [-6.621295529420721,   4.174826949593883,  -5.397414750887257,   4.107372047196714,   4.554937421642614,   -5.76491522658397,   4.391173394717607,   4.450525199056153,  -5.737414862957012,   4.378752054066173]

def sigmoid(x):
    x = (x if x >= -50 else -50) if x <= 50 else 50;
    return 1. / (1. + np.exp(-x));


def logloss(p, y):
    return -y * np.log(p) - (1. - y) * np.log(1. - p)



def select_threshold(threshhold):
    F = 14
    N = 200
    E = 0.

    for f in range(F):
        fin = open(r'C:\Temp\kaggle\CTR\data\lr_train\lr_train.b.%d' % f, "rb")
        for i in range(N):
            a = array('f')
            a.fromfile(fin, 10)

            y = a[0]
            a[0] = 1.

            a[1] = np.sqrt(a[1]);
            a[4] = np.sqrt(a[4]);
            a[7] = np.sqrt(a[7]);


            r = sigmoid(np.dot(theta, a))
            r = 1 if r > threshhold else 0
            E += abs(r - y)
        fin.close()

    return E/(N*(F-1)), threshhold


def calc_cost():
    F = 14
    N = 100
    E = 0.

    for f in range(F):
        fin = open(r'C:\Temp\kaggle\CTR\data\lr_train\lr_train.b.%d' % f, "rb")
        for i in range(N):
            a = array('f')
            a.fromfile(fin, 10)

            y = a[0]
            a[0] = 1.

            a[1] = np.sqrt(a[1]);
            a[4] = np.sqrt(a[4]);
            a[7] = np.sqrt(a[7]);


            r = sigmoid(np.dot(theta, a))
            E += logloss(r, y)
            #print r, y
        fin.close()

    return E/(N*(F-1))

def main():
    res = []
    for t in sp.linspace(0., 1., 500):
        res.append(select_threshold(t))
    res.sort()
    print res[:5]

    print "E:", calc_cost()



if __name__ == '__main__':
    main()
