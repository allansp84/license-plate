import os
import cv2
from ..utils import *
from ..localization.localization import locatePlate
from ..segmentation import Segmentation
from ..classification import SVMClassifier

import pdb


class Testing(object):

    def __init__(self, image_filename):

        self.output_path = '{0}/models/classifiers/svm'.format(os.getcwd())

        self.image = []
        self.image_filename = image_filename

        self.point = []

    def execute(self):

        letter_classes = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M',
                                   'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

        # -- Loading images
        img = cv2.imread(self.image_filename, cv2.CV_LOAD_IMAGE_COLOR)
        img = cv2.cvtColor(img,cv2.cv.CV_BGR2HSV)[:, :, 2]

        self.point = np.array(locatePlate(img, self.image_filename))

        # -- Segmentation of the characters
        self.seg = Segmentation(input_fname=self.image_filename, image=img, points=self.point, persist=False)
        self.seg.run()

        # -- Classification
        neg_idxs = np.array([idx for idx in xrange(len(self.seg.feature_vectors)) if (self.seg.feature_vectors == 0.0).all()])
        # is_plate = np.array([False if ((feat == 0.0).all()) else True for feat in self.seg.feature_vectors])

        svm_letter = SVMClassifier(model_fname='letters.model', output_path=self.output_path)
        svm_letter.load_model()

        letter_pred = svm_letter.model.predict(svm_letter.scaling_data_from_file(self.seg.feature_vectors[:3, :]))

        svm_number = SVMClassifier(model_fname='numbers.model', output_path=self.output_path)
        svm_number.load_model()
        number_pred = svm_number.model.predict(svm_number.scaling_data_from_file(self.seg.feature_vectors[3:, :]))

        if len(neg_idxs):
            letter_pred[:] = 0
            number_pred[:] = 0

        predicted = [letter_classes[int(l)] for l in letter_pred]
        predicted = ''.join(predicted)
        predicted += ''.join(number_pred.astype(np.str))

        if predicted == 'AAA0000':
            print 'None'

        else:
            print '{0},{1},{2},{3},{4},{5},{6},{7},{8}'.format(self.point[0,0],self.point[0,1],
                                                               self.point[1,0],self.point[1,1],
                                                               self.point[2,0],self.point[2,1],
                                                               self.point[3,0],self.point[3,1],
                                                               predicted)
