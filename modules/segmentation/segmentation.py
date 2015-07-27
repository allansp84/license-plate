import os
import cv2
import numpy as np

from skimage import util
from skimage import measure
from skimage import feature

from operator import itemgetter
from matplotlib import pyplot as plt

import pdb

class Segmentation(object):
    """
    Plate segmentation
    """
    def __init__(self, input_fname, image=0, points=0, dataset_path='./dataset', output_path='./working',
                 feature='hog', persist=True):

        self.input_fname = input_fname
        self.image = np.uint8(0)
        self.points = np.float32(points)
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.plate_number = 'AAA0000'
        self.plate_shape = (50, 200)
        self.plate_image = np.zeros(self.plate_shape)
        self.debug = False
        self.characters = []
        self.character_shape = (32, 32)
        self.feature = feature
        self.feature_vectors = []
        self.feature_fnames = []
        self.persist = persist

        rel_fname = os.path.relpath(self.input_fname, self.dataset_path)
        rel_fname = os.path.splitext(rel_fname)[0]
        self.output_fname = os.path.join(self.output_path, rel_fname)



    def load_data(self):

        self.image = cv2.imread(self.input_fname, cv2.CV_LOAD_IMAGE_COLOR)
        self.image = cv2.cvtColor(self.image,cv2.cv.CV_BGR2HSV)[:, :, 2]

    def get_points(self):

        filename_txt = os.path.splitext(self.input_fname)[0] + ".txt"
        data = np.loadtxt(filename_txt, dtype=np.str, delimiter=',')

        # -- hack to avoid temporarily the processing of two plate in the same image
        if len(data.shape) == 2:
            data = data[0]

        point_0 = np.array(data[0:2], dtype=np.float32)
        point_1 = np.array(data[2:4], dtype=np.float32)
        point_2 = np.array(data[4:6], dtype=np.float32)
        point_3 = np.array(data[6:8], dtype=np.float32)
        self.points = np.array([point_0, point_1, point_2, point_3])

    def get_plate_number(self):

        filename_txt = os.path.splitext(self.input_fname)[0] + ".txt"
        data = np.loadtxt(filename_txt, dtype=np.str, delimiter=',')

        # -- hack to avoid temporarily the processing of two plate in the same image
        if len(data.shape) == 2:
            data = data[0]

        data = np.char.strip(np.char.upper(data))

        if (data != 'NONE').all():
            self.plate_number = str(data[8]).upper()

    def sliding_concentric_windows(self, img):

        n_rows, n_cols = img.shape[:2]

        y_a, x_a = 5, 5
        y_b, x_b = 11, 11

        new_img = np.zeros((n_rows, n_cols), dtype=np.uint)
        img_a = util.pad(img, ((y_a/2, y_a/2), (x_a/2, x_a/2)), mode='constant')
        img_b = util.pad(img, ((y_b/2, y_b/2), (x_b/2, x_b/2)), mode='constant')

        blocks_a = util.view_as_windows(img_a, (y_a, x_a), step=1)
        blocks_b = util.view_as_windows(img_b, (y_b, x_b), step=1)

        for row in xrange(n_rows):
            for col in xrange(n_cols):
                mean_a = blocks_a[row, col].mean()
                mean_b = blocks_b[row, col].mean()

                r_mean = 0
                if mean_a != 0:
                    r_mean = mean_b/float(mean_a)

                if r_mean > 1.0:
                    new_img[row, col] = 0
                else:
                    new_img[row, col] = 255

        return new_img

    def normalization(self):
        n_rows, n_cols = self.plate_shape
        points_dst = np.array([[0, 0], [n_cols, 0], [n_cols, n_rows]], dtype=np.float32)
        warp_matrix = cv2.getAffineTransform(self.points[:3, :], points_dst)
        self.plate_image = cv2.warpAffine(self.image, warp_matrix, (n_cols, n_rows))

    def preprocessing(self):

        self.bimage = self.sliding_concentric_windows(self.plate_image)
        self.bimage = 255 - self.bimage
        self.bimage = np.uint8(self.bimage)

        kernel = np.ones((3,3), np.uint8)
        self.bimage = cv2.morphologyEx(self.bimage, cv2.MORPH_OPEN, kernel, iterations=1)

        kernel = np.ones((2,1), np.uint8)
        self.bimage = cv2.erode(self.bimage, kernel, iterations=1)
        self.bimage = cv2.dilate(self.bimage, kernel, iterations=2)

        thr, self.bimage = cv2.threshold(self.bimage, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        return True

    def find_bounding_box(self, contour):
        min_x, max_x = contour[:, 0].min(), contour[:, 0].max()
        min_y, max_y = contour[:, 1].min(), contour[:, 1].max()
        width = max_x - min_x
        height = max_y - min_y
        return np.array([min_x, min_y, width, height])

    def is_character(self, ratio, area, height):
        r_min, r_max = 1.0, 20.0
        pa_min, pa_max = 4, 20.0
        h_min, h_max = 25, 40
        is_char = False

        if (ratio > r_min) and (ratio < r_max):
            if (area > pa_min) and (area < pa_max):
                is_char = True
            else:
                if height > h_min:
                    is_char = True
                else:
                    is_char = False
        else:
            is_char = False

        return is_char

    def connected_component_analysis(self):

        labels = measure.label(self.bimage)
        label_number = 0

        results = []
        while True:
            temp = np.uint8(labels == label_number) * 255
            if not cv2.countNonZero(temp):
                break
            results.append(temp)
            label_number += 1
        results = np.array(results)

        img = np.zeros(self.bimage.shape, np.uint8)
        img += 255

        total_area = float(self.bimage.shape[0]*self.bimage.shape[1])

        self.new_img = np.zeros(self.bimage.shape, dtype=np.uint8)

        db = []
        for res in results:
            contours, hierarchy = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.shape[0] > 4:
                    contour = np.squeeze(contour)
                    bbox = self.find_bounding_box(contour)

                    aspect = (bbox[3]/float(bbox[2])) if float(bbox[2]) != 0 else 0
                    bbox_area = float(bbox[2]*bbox[3])
                    area_rate = (bbox_area/total_area) * 100.0

                    db.append([aspect, area_rate, bbox_area, bbox, contour])

        db = sorted(db, key=itemgetter(1), reverse=True)
        border = 2
        idx = 0
        found_all_characters = False
        while (idx < len(db)) and not found_all_characters:

            aspect, area_rate, bbox_area, bbox, contour = db[idx]

            if self.is_character(aspect, area_rate, bbox[3]):
                area_character = self.new_img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]].sum()
                intersection = (area_character/bbox_area)*100
                if intersection < 20.0:
                    self.new_img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1
                    if ((bbox[1]-border < 0) or (bbox[1]+bbox[3]+border > self.bimage.shape[0]) or \
                        (bbox[0]-border < 0) or (bbox[0]+bbox[2]+border > self.bimage.shape[1])):

                        self.characters.append([bbox[0], self.plate_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], contour])

                    else:

                        self.characters.append([bbox[0], self.plate_image[bbox[1]-border:bbox[1]+bbox[3]+border,
                                                                     bbox[0]-border:bbox[0]+bbox[2]+border], contour])

                    if self.debug:
                        print "(PLATE) R={0} PA={1} H={2}".format(aspect, area_rate, bbox[3])
                        cv2.rectangle(self.plate_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(0, 255, 0), thickness=1)
                        # plt.imshow(self.plate_image, cmap='gray'), plt.show()
                else:
                    if self.debug:
                        print "(NONPLATE) R={0} PA={1} H={2}".format(aspect, area_rate, bbox[3])
                        cv2.rectangle(self.plate_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(255, 0, 0), thickness=1)
                        # plt.imshow(self.plate_image, cmap='gray'), plt.show()

            else:
                if self.debug:
                    print "(NONPLATE) R={0} PA={1} H={2}".format(aspect, area_rate, bbox[3])
                    cv2.rectangle(self.plate_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(255, 0, 0), thickness=1)
                    # plt.imshow(self.plate_image, cmap='gray'), plt.show()

            idx += 1

            if len(self.characters) >= 7:
                found_all_characters = True

        self.characters = sorted(self.characters, key=itemgetter(0), reverse=False)
        if len(self.characters) != 7:
            if (len(self.characters) == 6) and ((self.characters[3][0] - self.characters[2][0]) > 50):
                self.characters.append([95, self.plate_image[2:48, 95:120], []])
                self.characters = sorted(self.characters, key=itemgetter(0), reverse=False)
                # found_all_characters = True
            else:
                self.plate_image = np.zeros(self.plate_shape)
                self.characters = []
                self.characters.append([0, self.plate_image[2:48, 0:25]])
                self.characters.append([1, self.plate_image[2:48, 25:50]])
                self.characters.append([2, self.plate_image[2:48, 50:75]])
                self.characters.append([3, self.plate_image[2:48, 95:120]])
                self.characters.append([4, self.plate_image[2:48, 120:145]])
                self.characters.append([5, self.plate_image[2:48, 145:170]])
                self.characters.append([6, self.plate_image[2:48, 170:195]])
                # found_all_characters = False

    def extract_features(self, persist=True):

        for idx in xrange(len(self.characters)):
            self.feature_fnames.append("{0}/{1}/{2}/{3}.npy".format(self.output_fname, self.plate_number, idx, self.plate_number[idx]))
            img = cv2.resize(self.characters[idx][1], self.character_shape)

            if self.feature == 'hog':
                self.feature_vectors.append(feature.hog(img, orientations=4, pixels_per_cell=(8, 8), cells_per_block=(1, 1)))

            elif self.feature == 'lbp':
                lbp = feature.local_binary_pattern(img, P=8, R=1, method='default')
                n_bins = lbp.max() + 1
                self.feature_vectors(np.histogram(lbp.ravel(), bins=n_bins, range=(0,n_bins), normed=True)[0])

            elif self.feature == 'daisy':
                self.feature_vectors(feature.daisy(img, step=4, radius=15, rings=3, histograms=8, orientations=8, normalization='l1', sigmas=None, ring_radii=None, visualize=False))

            else:
                threshold, feats = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                self.feature_vectors.append(feats.reshape(1,-1))

        self.feature_fnames = np.array(self.feature_fnames)
        self.feature_vectors = np.array(self.feature_vectors)

    def save_features(self):

        for idx in xrange(len(self.feature_vectors)):

            try:
                os.makedirs(os.path.dirname(self.feature_fnames[idx]))
            except OSError:
                pass

            np.save(self.feature_fnames[idx], self.feature_vectors[idx].reshape(1,-1))

    def run(self):

        # if os.path.exists(self.output_fname):
        #     return True

        if not self.image:
            self.load_data()

        if not (self.points == 0.0).all():
            self.get_plate_number()
            self.normalization()
            self.preprocessing()
            self.connected_component_analysis()
        else:
            self.plate_image = np.zeros(self.plate_shape)
            self.characters = []
            self.characters.append([0, self.plate_image[2:48, 0:25]])
            self.characters.append([1, self.plate_image[2:48, 25:50]])
            self.characters.append([2, self.plate_image[2:48, 50:75]])
            self.characters.append([3, self.plate_image[2:48, 95:120]])
            self.characters.append([4, self.plate_image[2:48, 120:145]])
            self.characters.append([5, self.plate_image[2:48, 145:170]])
            self.characters.append([6, self.plate_image[2:48, 170:195]])

        self.extract_features()

        if self.persist:
            self.save_features()

        return True
