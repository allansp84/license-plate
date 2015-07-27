# ------------------------------------------------------------

import os
import re
import cv2

import numpy as np

from skimage import transform
from skimage import filters

from sklearn import svm
from sklearn import grid_search
from sklearn import externals

# ------------------------------------------------------------

window_width = 90
window_height = 30


# ------------------------------------------------------------

def listFiles(path):
    file_list = []
    regex = re.compile(".*\.jpg")
    for root, sub_folders, files in os.walk(path):
        for f in files:
            if (regex.match(f)):
                file_list.append(f)
    return sorted(file_list)


# ------------------------------------------------------------

def splitDataset(dataset_path, files):
    plate_t = []
    plate_f = []

    for file_name in files:
        file_path = os.path.join(dataset_path, file_name)
        annotation = np.loadtxt(file_path.replace('.jpg', '.txt'), dtype=str, delimiter=',')
        if annotation.ndim == 1:
            plate_t.append(file_name)
        else:
            plate_f.append(file_name)

    return [plate_t, plate_f]


# ------------------------------------------------------------

def getPlateWindow(img, x, y):
    img_h = img.shape[0]
    img_w = img.shape[1]

    margin = 0
    x_min = np.amin(x) - margin
    x_max = np.amax(x) + margin
    y_min = np.amin(y) - margin
    y_max = np.amax(y) + margin

    x_ini = 0 if x_min < 0 else x_min
    y_ini = 0 if y_min < 0 else y_min
    x_end = img_w if x_max > img_w else x_max
    y_end = img_h if y_max > img_h else y_max

    w = x_end - x_ini
    h = y_end - y_ini
    return [x_ini, y_ini, w, h]


# ------------------------------------------------------------

def getRandomWindow(img):
    np.random.seed(7)
    h = img.shape[0] - window_height
    w = img.shape[1] - window_width
    y_ini = np.random.randint(h, size=1)
    x_ini = np.random.randint(w, size=1)
    return [x_ini, y_ini, window_width, window_height]


# ------------------------------------------------------------

def getFeatures(img, px, py, w, h):
    img_window = img[py: py + h, px: px + w]
    img_window = transform.resize(img_window, (window_height, window_width))
    img_window = np.absolute(filters.prewitt_v(img_window))
    return img_window.ravel()


# ------------------------------------------------------------

def extractFeatures(dataset_path, files):
    all_features = []

    for file_name in files:
        print file_name
        file_path = os.path.join(dataset_path, file_name)
        img = cv2.imread(file_path, 0)
        annotation = np.loadtxt(file_path.replace('.jpg', '.txt'), dtype=str, delimiter=',')

        if annotation.ndim == 1:
            x = [int(annotation[0]), int(annotation[2]), int(annotation[4]), int(annotation[6])]
            y = [int(annotation[1]), int(annotation[3]), int(annotation[5]), int(annotation[7])]
            px, py, w, h = getPlateWindow(img, x, y)
            features = getFeatures(img, px, py, w, h)
            all_features.append(features.tolist() + [1])
        else:
            for i in range(4):
                px, py, w, h = getRandomWindow(img)
                features = getFeatures(img, px, py, w, h)
                all_features.append(features.tolist() + [-1])

    return np.array(all_features)


# ------------------------------------------------------------

def trainModel(x_train, y_train):
    # @PARAMETER: grid search parameters

    log2c = np.logspace(-5,  20, 5, base=2).tolist()
    log2g = np.logspace(-15, 5, 5, base=2).tolist()

    tuned_parameters = [
        {
            'kernel': ['linear','rbf'],
            'gamma': log2g,
            'C': log2c,
        },
    ]

    clf = grid_search.GridSearchCV(svm.SVC(random_state=7), tuned_parameters, scoring='roc_auc', cv=5, verbose=10, n_jobs=5)
    clf.fit(x_train, y_train)
    print ''
    print 'with these params ' + str(clf.best_params_) + ' we got accuracy of ' + str(clf.best_score_)
    return clf


# ------------------------------------------------------------
# ------------------------------------------------------------

current_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(current_path, "train")

files = listFiles(dataset_path)
t, f = splitDataset(dataset_path, files)

pos = extractFeatures(dataset_path, t[0:300])
neg = extractFeatures(dataset_path, f[0:180])
data = np.vstack((pos, neg))

x_train = data[:, 0:-1]
y_train = data[:, -1]

clf = trainModel(x_train, y_train)
externals.joblib.dump(clf, os.path.join(current_path, "localization", "models", "svm.pkl"))
