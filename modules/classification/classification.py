import os
import time
import numpy as np

import matplotlib.pyplot as plt

from operator import itemgetter
from .svm import SVMClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix

import pdb


class Classification(object):

    def __init__(self, output_path, metafeat,
                 output_model=None,
                 file_type="npy",
                 algo='svm',
                 seed=42,
                 ):

        self.output_path = output_path
        self.metafeat = metafeat
        self.output_model = output_model
        self.file_type = file_type
        self.algo = algo
        self.seed = seed
        self.debug = False

    @property
    def dataset_path(self):
        return self.__dataset_path

    @dataset_path.setter
    def dataset_path(self, path):
        self.__dataset_path = os.path.abspath(path)

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, path):
        self.__output_path = os.path.abspath(path)

    def __load_features(self, fnames):
        X = []
        for i, fname in enumerate(fnames):
            if 'npy' in self.file_type:
                X += [np.load(fname)]
        X = np.array(X, dtype=np.float32)

        return np.reshape(X, ((-1,) + X.shape[2:]))

    def __load_all_features(self, fnames, labels, idxs):
        return labels[idxs], self.__load_features(fnames[idxs])

    def plot_confusion_matrix(self, y_true, y_pred, list_classes, title='Confusion matrix', filename='confusion_matrix.png'):

        # compute confusion matrix
        cm = confusion_matrix(y_true,y_pred)

        conf_mat_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        conf_mat2 = np.around(conf_mat_norm,decimals=2) # rounding to display in figure

        fig = plt.figure(figsize=(16,16), dpi=100)
        plt.imshow(conf_mat2,interpolation='nearest')

        for x in xrange(len(list_classes)):
          for y in xrange(len(list_classes)):
            plt.annotate(str(conf_mat2[x][y]),xy=(y,x),ha='center',va='center')

        plt.xticks(range(len(list_classes)),list_classes,rotation=90,fontsize=11)
        plt.yticks(range(len(list_classes)),list_classes,fontsize=11)

        plt.tight_layout(pad=3.)

        plt.title(title)
        plt.colorbar()

        # plt.show()
        fig.savefig(filename)

    def performance_eval(self, fnames_letters, fnames_numbers, letter_classes,
                         number_classes, letter_pred, letter_gt, number_pred, number_gt):

        letter_db = []
        for idx, fname in enumerate(fnames_letters):
            pos = int(fname.split('/')[-2])
            plate_number = fname.split('/')[-3]
            id_sample = fname.split('/')[-4]
            letter_db.append((id_sample, pos, letter_pred[idx], letter_gt[idx]))

        number_db = []
        for idx, fname in enumerate(fnames_numbers):
            pos = int(fname.split('/')[-2])
            plate_number = fname.split('/')[-3]
            id_sample = fname.split('/')[-4]
            number_db.append((id_sample, pos, number_pred[idx], number_gt[idx]))

        letter_db = np.array(sorted(letter_db, key=itemgetter(0), reverse=False))
        number_db = np.array(sorted(number_db, key=itemgetter(0), reverse=False))

        plate_pred = []
        plate_gt = []

        samples = np.unique(letter_db[:,0])
        for sample in samples:
            plate_idxs = [idx for idx in xrange(len(fnames_letters)) if sample in fnames_letters[idx]]
            plate = letter_db[plate_idxs]
            plate = np.array(sorted(plate, key=itemgetter(1), reverse=False))

            predicted = [letter_classes[int(l)] for l in plate[:, 2]]
            predicted = ''.join(predicted)

            grounth_truth = [letter_classes[int(l)] for l in plate[:, 3]]
            grounth_truth = ''.join(grounth_truth)

            plate_pred.append(predicted)
            plate_gt.append(grounth_truth)

        samples = np.unique(number_db[:,0])
        for i, sample in enumerate(samples):
            plate_idxs = [idx for idx in xrange(len(fnames_numbers)) if sample in fnames_numbers[idx]]
            plate = number_db[plate_idxs]
            plate = np.array(sorted(plate, key=itemgetter(1), reverse=False))

            predicted = [number_classes[int(l)] for l in plate[:, 2]]
            predicted = ''.join(predicted)

            grounth_truth = [number_classes[int(l)] for l in plate[:, 3]]
            grounth_truth = ''.join(grounth_truth)

            plate_pred[i] += predicted
            plate_gt[i] += grounth_truth

        plate_pred = np.array(plate_pred)
        plate_gt = np.array(plate_gt)

        return plate_pred, plate_gt

    def testing(self, classifier, test_set):

        print '-- classifying test set'

        classifier.test_set = test_set
        outputs = classifier.testing()

        gt = outputs['ground_truth']
        pred = outputs['predictions']

        acc = (pred == gt).sum() / float(test_set['data'].shape[0])

        return acc, outputs['predictions'], outputs['ground_truth']

    def classification(self, all_fnames, all_labels, train_idxs, test_idxs, idxs, model_fname='svms.model'):

        # -- building a classifier for letters
        new_train_idxs = np.intersect1d(train_idxs, idxs)
        train_labels, train_data = self.__load_all_features(all_fnames, all_labels, new_train_idxs)
        train_set = {'data': train_data, 'labels': train_labels}

        new_test_idxs = np.intersect1d(test_idxs, idxs)
        test_labels, test_data = self.__load_all_features(all_fnames, all_labels, new_test_idxs)
        test_set = {'data':test_data, 'labels': test_labels}

        neg_idxs = np.array([idx for idx, data in enumerate(test_set['data']) if (data == 0.0).all()])

        svm_clf = SVMClassifier(model_fname=model_fname, output_path=self.output_path)
        svm_clf.train_set = train_set
        svm_clf.training()

        acc, letter_pred, letter_gt = self.testing(svm_clf, test_set)

        letter_pred[neg_idxs] = 0

        return acc, letter_pred, letter_gt


    def running(self):

        try:
            os.makedirs(self.output_path)
        except Exception, e:
            pass

        all_fnames = self.metafeat['all_fnames']
        all_labels = self.metafeat['all_labels']
        train_idxs = self.metafeat['train_idxs']
        test_idxs = self.metafeat['test_idxs']
        all_letters_idxs = self.metafeat['all_letters_idxs']
        all_numbers_idxs = self.metafeat['all_numbers_idxs']

        letter_classes = np.unique(all_labels[all_letters_idxs])
        number_classes = np.unique(all_labels[all_numbers_idxs])

        neg_idxs = np.array([idx for idx, fname in enumerate(all_fnames) if ('AAA0000' in fname)])
        train_idxs = np.setdiff1d(train_idxs, neg_idxs)

        start = time.time()

        acc, letter_pred, letter_gt = self.classification(all_fnames, all_labels, train_idxs, test_idxs, all_letters_idxs,
                                                    model_fname='letters.model')

        acc, number_pred, number_gt = self.classification(all_fnames, all_labels, train_idxs, test_idxs, all_numbers_idxs,
                                                    model_fname='numbers.model')

        fnames_letters = np.intersect1d(all_fnames[test_idxs], all_fnames[all_letters_idxs])
        fnames_numbers = np.intersect1d(all_fnames[test_idxs], all_fnames[all_numbers_idxs])

        plate_pred, plate_gt = self.performance_eval(fnames_letters, fnames_numbers, letter_classes, number_classes,
                                                     letter_pred, letter_gt, number_pred, number_gt)

        print "AVERAGE: {0}".format(np.sum(plate_pred == plate_gt)/float(len(plate_pred)))

        elapsed = time.strftime("%j,%H,%M,%S",time.gmtime((time.time() - start))).split(',')
        print "elapsed time: {0} days and {1}h{2}m{3}s".format(int(elapsed[0])-1,elapsed[1],elapsed[2],elapsed[3])

        self.plot_confusion_matrix(letter_gt, letter_pred, letter_classes,
                                   title='Confusion Matrix for Letter Classifier',
                                   filename='confusion_matrix_letter.png')
        self.plot_confusion_matrix(number_gt, number_pred, number_classes,
                                   title='Confusion Matrix for Numbers Classifier',
                                   filename='confusion_matrix_number.png')

        return True
