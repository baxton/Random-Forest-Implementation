

from array import array
import pathes


FEATURES_OLD = 25
FEATURES_NEW = 100      # 1 to 4

VEC_LEN = 32
VEC_LEN_NEW = 2 + FEATURES_NEW

CNT_IDX = 0
CLK_IDX = 1
RATE_IDX = 2

NULL = -999999999


def process(fname, hists):
    fname_out = fname + ".hist"

    with open(fname, 'rb') as fin:
        with open(fname_out, "wb+") as fout:
            try:
                while True:
                    a = array('l')
                    a.fromfile(fin, VEC_LEN)

                    o = array('l', [NULL] * VEC_LEN_NEW)

                    o[0] = a[0] # id
                    o[1] = a[1] # click

                    for f in range(FEATURES_OLD):
                        idx = 2 + f
                        val = a[idx]
                        o[idx] = val
                        o[idx + FEATURES_OLD] = hists[f][val][CNT_IDX]
                        o[idx + FEATURES_OLD * 2] = hists[f][val][CLK_IDX]
                        o[idx + FEATURES_OLD * 3] = hists[f][val][RATE_IDX]
                    o.tofile(fout)

            except Exception as ex:
                print "# ", ex, "EOF"




def main():
    fname = pathes.path_train + "train.csv.b"
    #fname = pathes.path_train + "test.csv.b"

    hists = [None] * FEATURES_OLD

    with open("hist_train.csv", 'r') as fhist:
        for line in fhist:
            line = line.strip()
            vals = line.split(',')

            f = int(vals[0])
            val = int(vals[1])
            appears = int(vals[2])
            clicks = int(vals[3])
            rate = int( float(vals[4]) * 10000000 )

            if None == hists[f]:
                hists[f] = dict()

            hists[f][val] = (appears, clicks, rate)

    process(fname, hists)


if __name__ == '__main__':
    main()
