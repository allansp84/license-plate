import os
import sys
import operator
import numpy as np
import itertools as it

from glob import glob
from multiprocessing import Pool

from .config import *

def start_process():
    pass

def do_something(d):
    result = d.run()
    return result

def retrieve_samples(input_path, file_type):

    dir_names = []
    for root, subFolders, files in os.walk(input_path):
        for f in files:
            if f[-len(file_type):] == file_type:
                dir_names += [root]
                break

    dir_names = sorted(dir_names)

    fnames = []
    for dir_name in dir_names:
        dir_fnames = sorted(glob(os.path.join(input_path, dir_name, '*.' + file_type)))
        fnames += dir_fnames

    return fnames

def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 10
    vector[-pad_width[1]:] = 10
    return vector


def grouper(n, iterable, fillvalue=None):
    """
    grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return it.izip_longest(fillvalue=fillvalue, *args)

def mosaic(w, imgs):
    """
    Make a grid from images.
    w    -- number of grid columns
    imgs -- images (must have same size and format)
    """
    imgs = iter(imgs)
    img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))

def progressbar(name, i, total, barLen = 20):
        percent = float(i) / total

        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("%s: [ %s ] %.2f%%" % (name, progress, percent * 100))
        sys.stdout.flush()

class RunInParallel():
    def __init__(self, tasks, n_proc =(cpu_count()-1)):
        self._pool = Pool(initializer=start_process, processes=n_proc)
        self._tasks = tasks

    def run(self):
        pool_outs = self._pool.map_async(do_something, self._tasks)
        self._pool.close()
        self._pool.join()

        try:
            work_done = [out for out in pool_outs.get() if out==True ]
            assert (len(work_done)) == len(self._tasks)
        except Exception:
            sys.stderr.write("ERROR: some objects could not be processed!\n")
            sys.exit(1)

def replace_from_list(string_list, old_str, new_str):
    return map(lambda x: str.replace(x, old_str, new_str), string_list)

def creating_csv(f_results, output_path, test_set, measure):

    f_measure  = [f for f in f_results if((test_set in f) and (measure in f))]

    configs, values = [], []
    for f_m in f_measure:
        configs += [os.path.dirname(os.path.relpath(f_m, output_path))]
        values += [float(open(f_m,'r').readline())]

    configs_orin = configs
    configs = replace_from_list(configs, '/', ',')

    configs = replace_from_list(configs, test_set, '')
    configs = replace_from_list(configs, 'classifiers,', '')

    configs = replace_from_list(configs, '300,', '')

    configs = replace_from_list(configs, 'realization_1', 'R1')
    configs = replace_from_list(configs, 'realization_2', 'R2')
    configs = replace_from_list(configs, 'realization_3', 'R3')

    configs = replace_from_list(configs, 'centerframe', 'C')
    configs = replace_from_list(configs, 'wholeframe', 'W')

    configs = replace_from_list(configs, 'dftenergymag',   'ME')
    configs = replace_from_list(configs, 'dftentropymag',  'MS')
    configs = replace_from_list(configs, 'dftenergyphase', 'PE')
    configs = replace_from_list(configs, 'dftentropyphase','PS')

    configs = replace_from_list(configs, 'kmeans', 'K')
    configs = replace_from_list(configs, 'random', 'R')

    configs = replace_from_list(configs, 'class_based', 'D')
    configs = replace_from_list(configs, 'unified', 'S')

    configs = replace_from_list(configs, 'svm', 'SVM')
    configs = replace_from_list(configs, 'pls', 'PLS')

    configs = replace_from_list(configs, 'energy_phase', 'PE')
    configs = replace_from_list(configs, 'entropy_phase', 'PH')
    configs = replace_from_list(configs, 'energy_mag', 'ME')
    configs = replace_from_list(configs, 'entropy_mag', 'MH')
    configs = replace_from_list(configs, 'mutualinfo_phase', 'PMI')
    configs = replace_from_list(configs, 'mutualinfo_mag', 'MMI')
    configs = replace_from_list(configs, 'correlation_phase', 'PC')
    configs = replace_from_list(configs, 'correlation_mag', 'MC')

    reverse = False if 'hter' in measure else True

    results = sorted(zip(configs, values), key=operator.itemgetter(1), reverse=reverse)

    fname = "{0}/{1}.{2}.csv".format(output_path, test_set, measure)
    f_csv = open(fname, 'w')
    f_csv.write("N,LGF,M,CS,SDD,DS,CP,C,%s\n" % str(measure).upper())
    for r in results:
        f_csv.write("%s%s\n" % (r[0], r[1]))
    f_csv.close()

    print fname, results[:4]

    return sorted(zip(configs_orin, values), key=operator.itemgetter(1), reverse=reverse)
