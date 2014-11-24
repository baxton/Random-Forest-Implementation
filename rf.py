############################################################
#
#
#
############################################################


import os
import sys
import numpy as np
import scipy as sp
import ctypes

import pathes
import utils




RF_DLL = ctypes.cdll.LoadLibrary(pathes.path_base + "scripts/sub1/rf.dll")




class RF(object):

    def __init__(self, columns, trees_num=10, nfeatures=5, samples_per_leaf=5):
        self.trees_num = trees_num
        self.nfeatures = nfeatures
        self.samples_per_leaf = samples_per_leaf

        self.trees = []

        for t in range(self.trees_num):
            ptree = RF_DLL.alloc_tree_learner(ctypes.c_int(columns), ctypes.c_int(self.nfeatures), ctypes.c_int(self.samples_per_leaf))
            self.trees.append(ptree)
    # END of ctor

    def tofile(self, fname):
        for t in range(self.trees_num):
            fn = fname + str(t) + ".rf"
            cfn = ctypes.c_char_p(fn)
            RF_DLL.tofile_tree(ctypes.c_void_p(self.trees[t]), cfn)

    def release(self):
        for t in range(self.trees_num):
            RF_DLL.free_tree(ctypes.c_void_p(self.trees[t]))

        self.trees = []
    # END of release

    def prepare_bootstraps(self, num_in_bootstrap, total_num):
        self.bootstraps = []
        for t in range(self.trees_num):
            self.bootstraps.append( {} )
            for i in sp.random.randint(0, total_num, num_in_bootstrap):
                if i in self.bootstraps[t]:
                    self.bootstraps[t][i] += 1
                else:
                    self.bootstraps[t][i] = 1
    # END of prepare_bootstraps

    def start_fit(self):
        for t in range(self.trees_num):
            RF_DLL.start_fit_tree(ctypes.c_void_p(self.trees[t]))


    def fit(self, id, x, y):
        for t in range(self.trees_num):
            if id in self.bootstraps[t]:
                for i in range(self.bootstraps[t][id]):
                    RF_DLL.fit_tree(ctypes.c_void_p(self.trees[t]), x.ctypes.data, ctypes.c_double(y))



    def stop_fit(self):
        for t in range(self.trees_num):
            RF_DLL.stop_fit_tree(ctypes.c_void_p(self.trees[t]))



    def is_finished(self):
        cnt = 0
        for t in range(self.trees_num):
            cnt += RF_DLL.end_of_splitting(ctypes.c_void_p(self.trees[t]))

        return cnt == self.trees_num


    def predict(self, x):
        p = 0.
        for t in range(self.trees_num):
            tmp_p = ctypes.c_double(0)
            RF_DLL.predict(ctypes.c_void_p(self.trees[t]), x.ctypes.data, ctypes.addressof(tmp_p))
            p += tmp_p.value
        return p / self.trees_num


    def predict_proba(self, x):
        p = self.predict(x)
        return np.array([ 1. - p, p ])



    def print_rf(self):
        RF_DLL.print_tree(ctypes.c_void_p(self.trees[0]))


