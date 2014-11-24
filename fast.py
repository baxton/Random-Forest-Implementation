

import numpy as np
import scipy as sp

from datetime import datetime
from csv import DictReader

import pathes
import utils
import rf_sparse



train_fname = pathes.path_train + 'train.csv'
test_fname = pathes.path_train + 'test.csv'






##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()


TOTAL_SPARSE_FEATURES = 9449205
TOTAL_FEATURES        = (4 + TOTAL_SPARSE_FEATURES)
SPARSE_PART_SIZE      = 21
F_IDX                 = 4

VEC_LEN = 4 + SPARSE_PART_SIZE

N = utils.TRAIN_RECORDS_NUM
NBS = 500000


learner = rf_sparse.RF(TOTAL_FEATURES, trees_num=10, nfeatures=3500, samples_per_leaf=5)
learner.prepare_bootstraps(NBS, N)
learner.start_fit()



# start training
cnt = 0
while True:

    pass_start = datetime.now()

#    for t, ID, x, y in data(train_fname, D):  # data is a generator
    cnt = 0
    print "# read", cnt, "lines"

    with open(train_fname, 'r') as fin:
        for line in fin:
            cnt += 1
            if 0 == (cnt % 100000):
                print "# read", cnt, "lines"

            line = line.strip()
            #tokens = np.fromstring(line, sep=',')
            tokens = line.split(',')

            x = np.array(tokens[3:])
            #x = tokens[3:]

            y = float(tokens[1])
            ID = int(tokens[2])
            #y = tokens[1]
            #ID = tokens[2]

            learner.fit(ID, x, y)
            #
        learner.stop_fit()

    time_delta = datetime.now() - pass_start

    cnt += 1
    print "pass #", cnt, " time elapsed: ", str(time_delta)
    if learner.is_finished():
        break

    #break

learner.tofile(pathes.path_base + "rf_2_")

print "Done, time elapsed: ", str(datetime.now() - start)




##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################
##submission = pathes.path_base + 'scripts/sub1/submit01.txt'
##with open(submission, 'w') as outfile:
##    outfile.write('id,click\n')
##    for t, ID, x, y in data(test_fname, D):
##        p = learner.predict(x)
##        outfile.write('%s,%s\n' % (ID, str(p)))
