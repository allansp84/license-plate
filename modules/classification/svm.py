import os
import sys
import cPickle
import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.multiclass import OneVsRestClassifier
from scipy.ndimage import convolve
from sklearn import externals


class SVMClassifier(object):

    """docstring for SVMClassifier"""

    def __init__(self, model_fname='svms.model', output_path='./', scale_features=True, n_jobs=4, seed=42, debug=False):

        super(SVMClassifier, self).__init__()

        # private attributes
        self.__seed = seed
        self.__debug = debug
        self.fname_model = os.path.join(output_path, model_fname)
        self.fname_scaling_params = os.path.splitext(os.path.join(output_path, model_fname))[0] + '_scaling_params.txt'

        # public attributes
        self.output_path = output_path
        self.scale_features = scale_features
        self.n_jobs = n_jobs
        self.train_set = {}
        self.test_set = {}
        self.model = []
        self.scaling_params = {}


    @property
    def train_set(self):
        return self.__train_set

    @train_set.setter
    def train_set(self, train_dict):
        try:
            assert isinstance(train_dict, dict)

            if train_dict:
                self.__train_set = {'data':[], 'labels': [], 'is_scaled': False}

                self.__train_set['data'] = train_dict['data']
                self.__train_set['labels'] = train_dict['labels']

                self.scaling_params = {}

        except Exception, e:
            raise e

    @property
    def test_set(self):
        return self.__test_set

    @test_set.setter
    def test_set(self, test_dict):
        try:
            assert isinstance(test_dict, dict)

            if test_dict:
                self.__test_set = {'data':[], 'labels': [], 'is_scaled': False}

                self.__test_set['data'] = test_dict['data']
                self.__test_set['labels'] = test_dict['labels']

        except Exception, e:
            raise e

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model_list):
        try:
            # assert isinstance(model_list, list)
            self.__model = model_list
        except Exception, e:
            raise e

    @property
    def scaling_params(self):
        if self.__scaling_params:
            return self.__scaling_params
        else:
            self.__scaling_params = self.__compute_stats()
            return self.__scaling_params

    @scaling_params.setter
    def scaling_params(self, params):
        try:
            assert isinstance(params, dict)
            self.__scaling_params = params
        except Exception, e:
            raise e

    @property
    def scale_features(self):
        return self.__scale_features

    @scale_features.setter
    def scale_features(self, is_scale_features):
        try:
            assert isinstance(is_scale_features, bool)
            self.__scale_features = is_scale_features
        except Exception, e:
            raise e

    @property
    def output_path(self):
        return self.__output_path

    @output_path.setter
    def output_path(self, path):
        self.__output_path = os.path.abspath(path)

    def __save_model(self, model, fname):

        if self.__debug:
            print '-- saving model'

        try:
            os.makedirs(os.path.dirname(fname))
        except OSError:
            pass

        if not os.path.isfile(fname):
            fo = open(fname, 'wb')
            cPickle.dump(model,fo)
            fo.close()

            externals.joblib.dump(model, fname)


    def __one_svm(self, cat):

        lcat = np.zeros(self.train_set['labels'].size)

        lcat[self.train_set['labels'] != cat] = -1
        lcat[self.train_set['labels'] == cat] = +1

        # -- build set the parameters for grid search
        log2c = np.logspace(-5,  20, 16, base=2).tolist()
        log2g = np.logspace(-15, 5, 16, base=2).tolist()

        search_space = [{'kernel':['rbf'], 'gamma':log2g, 'C':log2c, 'class_weight':['auto']}]
        search_space += [{'kernel':['linear'], 'C':log2c, 'class_weight':['auto']}]

        svm = GridSearchCV(SVC(random_state=self.__seed), search_space, cv=2, scoring='roc_auc', n_jobs=self.n_jobs)
        # svm = SVC(random_state=self.__seed)

        svm.fit(self.train_set['data'], lcat)

        return svm

    def __compute_stats(self):

        params = {}
        if self.train_set:
            mean = self.train_set['data'].mean(axis=0)
            std = self.train_set['data'].std(axis=0, ddof=1)
            std[std == 0.] = 1.
            params = {'mean': mean, 'std': std}

        else:
            sys.exit('Train set not found!')

        return params

    def __scaling_data(self, scaled_data):

        params = self.scaling_params

        scaled_data -= params['mean']
        scaled_data /= params['std']

        np.savetxt(self.fname_scaling_params, np.array([params['mean'], params['std']]), fmt='%1.4f')

        return scaled_data

    def scaling_data_from_file(self, scaled_data):

        params = np.loadtxt(self.fname_scaling_params)
        scaled_data -= params[0]
        scaled_data /= params[1]

        return scaled_data

    def __load_persisted_model(self, fname):
        # fo = open(fname, 'rb')
        # model = cPickle.load(fo)
        # fo.close()
        return externals.joblib.load(fname)

    def load_model(self):

        model = []

        if self.__debug:
            print '-- loading model'

        # -- load model persisted on disk
        if os.path.isfile(self.fname_model):
            model = self.__load_persisted_model(self.fname_model)
            if self.__debug:
                print '-- found in {0}'.format(self.fname_model)

        # -- load model found in memory
        elif self.model:
            model = self.model
            if self.__debug:
                print '-- found in memory'

        # -- model does not generated yet
        else:
            if self.__debug:
                print '-- not found'

        self.model = model

    def nudge_dataset(self, X, Y):
        """
        This produces a dataset 5 times bigger than the original one,
        by moving the 8x8 images in X around by 1px to left, right, down, up
        """
        direction_vectors = [
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]]

        shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()

        X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
        Y = np.concatenate([Y for _ in range(5)], axis=0)

        return X, Y

    def nudge(self, X, y):
        # initialize the translations to shift the image one pixel
        # up, down, left, and right, then initialize the new data
        # matrix and targets
        translations = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        data = []
        target = []

        # loop over each of the digits
        for (image, label) in zip(X, y):
            # reshape the image from a feature vector of 784 raw
            # pixel intensities to a 28x28 'image'
            image = image.reshape(32, 32)

            # loop over the translations
            for (tX, tY) in translations:
                # translate the image
                M = np.float32([[1, 0, tX], [0, 1, tY]])
                trans = cv2.warpAffine(image, M, (32, 32))

                # update the list of data and target
                data.append(trans.flatten())
                target.append(label)

        # return a tuple of the data matrix and targets
        return (np.array(data), np.array(target))


    def training(self):

        print 'Training ...'

        # -- try loading model
        self.load_model()

        # -- True if model does not generated yest
        if not self.model:

            if self.__debug:
                print '-- building model'

            # -- True if train set doesn't scaled yest and if is to scale it
            if((self.scale_features) and (not self.train_set['is_scaled'])):
                self.train_set['data'] = self.__scaling_data(self.train_set['data'])
                self.train_set['is_scaled'] = True

            model = []
            # # -- compute model for eat category
            # categories = np.unique(self.train_set['labels'])
            # for cat in categories:
            #     model += [self.__one_svm(cat)]

            parameters = {
                "estimator__C": [1,2,4,8],
                "estimator__kernel": ["poly","rbf"],
                "estimator__degree":[1, 2, 3, 4],
            }


            # binarize the output
            y = [ord(x.lower())-97 if ord(x.lower())-96 > 0 else int(x) for x in self.train_set['labels']]
            # y = label_binarize(y, classes=range(len(np.unique(y))))

            # Learn to predict each class against the other
            model_to_set = OneVsRestClassifier(SVC(random_state=7, class_weight='auto'))
            model = GridSearchCV(model_to_set, param_grid=parameters, cv=2, score_func='roc_auc', verbose=10, n_jobs=4)
            model.fit(self.train_set['data'], y)

            print ''
            print 'with these params ' + str(model.best_params_) + ' we got accuracy of ' + str(model.best_score_)

            print self.fname_model
            self.__save_model(model, self.fname_model)

            self.model = model

        if self.__debug:
            print '-- finished'

    def testing(self):

        print 'Testing ...'

        outputs = {}

        self.load_model()

        if self.model:

            if self.scale_features and (not self.test_set['is_scaled']):
                self.test_set['data'] = self.__scaling_data(self.test_set['data'])
                self.test_set['is_scaled'] = True

            n_test = self.test_set['data'].shape[0]
            categories = np.unique(self.test_set['labels'])
            n_categories = len(categories)

            cat_index = {}
            predictions = np.empty((n_test, n_categories))

            model = self.model
            for icat, cat in enumerate(categories):
                cat_index[cat] = icat
                # resps = model[icat].decision_function(self.test_set['data'])
                # predictions[:, icat] = resps

            # predictions = model.decision_function(self.test_set['data'])
            # pred = predictions.argmax(axis=1)

            pred = model.predict(self.test_set['data'])

            gt = np.array([cat_index[e] for e in self.test_set['labels'].reshape(-1)]).astype('int')

            outputs = {'predictions': pred, 'ground_truth': gt}

        else:
            sys.exit('-- model not found! Please, execute training again!')

        return outputs

    def run(self):

        self.training()
        outputs = self.testing()

        return outputs
