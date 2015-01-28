from array import array
import scipy as sp

zeros = []
ones = []


def main():
    ID = 0
    for f in range(14):
        fin = open(r'C:\Temp\kaggle\CTR\data\lr_train\lr_train.b.%d' % f, "rb")
        try:
            while (True):
                a = array('f')
                a.fromfile(fin, 8)

                if a[0] == 1:
                    ones.append(ID)
                else:
                    zeros.append(ID)

                ID += 1

        except Exception as ex:
            print ex, "EOF"

    result = array('l')
    result.extend( sp.random.choice(zeros, 1000000) )
    result.extend( sp.random.choice(ones, 1000000) )
    result = array('l', np.sort(result))

    with open(r'C:\Temp\kaggle\CTR\cv_click_ids.b', "wb+") as fout:
        result.tofile(fout)



if __name__ == '__main__':
    main()
