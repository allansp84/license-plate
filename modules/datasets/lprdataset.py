import os
import itertools
import numpy as np
from glob import glob

import pdb

class LPRDataset(object):

    def __init__(self, dataset_path='./dataset', output_path='./working', file_types=['jpg']):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.file_types = file_types

    def _list_dirs(self, rootpath, filetype):
        folders = []

        for root, dirs, files in os.walk(rootpath):
            for f in files:
                if filetype in os.path.splitext(f)[1]:
                    folders += [os.path.relpath(root, rootpath)]
                    break

        return folders

    def split_dataset(self, all_fnames, training_rate=0.8):

        samples = []
        for fname in all_fnames:
            samples.append(fname.split('/')[-4])
        samples = np.unique(samples)

        n_train_samples = int(len(samples) * training_rate)

        samples_train = samples[:n_train_samples]
        samples_test = samples[n_train_samples:]

        train_idxs = []
        for sample in samples_train:
            train_idxs += [idx for idx in xrange(len(all_fnames)) if sample in all_fnames[idx]]

        test_idxs = []
        for sample in samples_test:
            test_idxs += [idx for idx in xrange(len(all_fnames)) if sample in all_fnames[idx]]

        all_idxs = np.concatenate((train_idxs, test_idxs)).astype(np.int)

        # rstate = np.random.RandomState(7)
        rand_idxs = all_idxs  # rstate.permutation(all_idxs)

        n_train_idxs = int(len(rand_idxs) * training_rate)

        return all_idxs, rand_idxs[:n_train_idxs], rand_idxs[n_train_idxs:]


    def _build_meta(self, inpath, filetypes):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []

        train_idxs = []
        test_idxs = []

        # folders = np.array(sorted(self._list_dirs(inpath, filetype)))
        folders = [self._list_dirs(inpath, filetype) for filetype in filetypes]
        # flat and sort list of fnames
        folders = itertools.chain.from_iterable(folders)
        folders = sorted(list(folders))

        for i, folder in enumerate(folders):
            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = itertools.chain.from_iterable(fnames)
            fnames = sorted(list(fnames))

            for fname in fnames:
                label = 0
                if len(os.path.relpath(fname, inpath).split('/')) == 2:
                    text_fname = "{0}.txt".format(os.path.splitext(fname)[0])
                    if os.path.isfile(text_fname):
                        data = np.loadtxt(text_fname, np.str, delimiter=',')
                        if data.shape:
                            label = 1

                elif len(os.path.relpath(fname, inpath).split('/')) == 5:
                    filename = os.path.relpath(fname, inpath).split('/')[-1]
                    label = os.path.splitext(filename)[0]

                if 'train/' in os.path.relpath(fname, inpath):
                    all_idxs += [img_idx]
                    train_idxs += [img_idx]
                    all_fnames += [fname]
                    all_labels += [label]
                    img_idx += 1
                else:
                    pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)

        all_idxs, train_idxs, test_idxs = self.split_dataset(all_fnames)

        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)
        all_idxs = np.array(all_idxs)

        numbers = np.array([str(i) for i in xrange(10)])
        letters = np.setdiff1d(all_labels, numbers)

        all_letters_idxs = []
        for lt in letters:
            all_letters_idxs += list(np.where(all_labels == lt)[0])
        all_letters_idxs = np.array(all_letters_idxs)

        all_numbers_idxs = []
        for nb in numbers:
            all_numbers_idxs += list(np.where(all_labels == nb)[0])
        all_numbers_idxs = np.array(all_numbers_idxs)

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'all_letters_idxs': all_letters_idxs,
                  'all_numbers_idxs': all_numbers_idxs,
                  'train_idxs': train_idxs,
                  'test_idxs': test_idxs,
                  }

        return r_dict

    @property
    def metainfo(self):
        try:
            return self.__metainfo
        except AttributeError:
            self.__metainfo = self._build_meta(self.dataset_path, self.file_types)
            return self.__metainfo

    def metainfo_feats(self, output_path, file_types):
        return self._build_meta(output_path, file_types)

    def metainfo_images(self, output_path, file_types):
        return self._build_meta(output_path, file_types)
