# ------------------------------------------------------------

import os
import numpy as np

from skimage import measure
from skimage import filters
from skimage import transform
from skimage import morphology
from skimage import img_as_ubyte

from sklearn.externals import joblib

# ------------------------------------------------------------

def resizeImage(img_orig):
    new_width = 640.
    new_height = 480.

    height = img_orig.shape[0]
    width = img_orig.shape[1]

    if (width > height):
        scale = new_width / width
        new_height = height * scale
    else:
        scale = new_height / height
        new_width = width * scale

    img_resized = transform.resize(img_orig, (int(new_height), int(new_width)))
    return [scale, img_resized]


# ------------------------------------------------------------

def getRegions(img_orig):
    # sizes = [4, 6, 8, 10]
    sizes = [4, 8, 12]

    img_processed = (255 * np.absolute(filters.prewitt_v(img_orig))).astype(np.uint8)
    img_aux = (255 * np.absolute(filters.prewitt_h(img_orig))).astype(np.uint8)
    img_processed[img_aux > 10] = 0

    regions_of_interest = []

    for s in sizes:
        img_processed = morphology.dilation(img_processed, morphology.square(s))
        thresh = filters.threshold_otsu(img_processed)
        img_processed = img_processed >= thresh
        labels = measure.label(img_processed)

        for region in measure.regionprops(labels):
            minr, minc, maxr, maxc = region.bbox
            ratio = float(maxr - minr) / float(maxc - minc)
            area = region.area
            if area > 1500 and area < 50000 and ratio < 1 and ratio > 0.2:
                regions_of_interest.append(region)

    return regions_of_interest


# ------------------------------------------------------------

def filterRegions(img_orig, regions):
    width = 90
    height = 30

    current_path = os.path.dirname(os.path.realpath(__file__))
    clf = joblib.load(os.path.join(current_path, "models", "svm.pkl"))

    if len(regions) > 0:
        scores = []

        for i, region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox
            img_window = img_orig[minr: maxr, minc: maxc]
            img_window = transform.resize(img_window, (height, width))
            img_window = np.absolute(filters.prewitt_v(img_window))
            features = img_window.ravel()
            scores.append((clf.decision_function(features)[0], i))

        # all scores
        scores.sort(reverse=True)
        scores = np.array(scores)
        i = int(scores[0, 1])
        return [regions[i]]

    return []


# ------------------------------------------------------------

def refineBoundaries(img_orig, plate):
    np.random.seed(7)

    minr, minc, maxr, maxc = plate.bbox
    img_window = img_orig[minr: maxr, minc: maxc]

    plate_points = [(minc, minr), (maxc, minr), (maxc, maxr), (minc, maxr)]
    plate_x = minc
    plate_y = minr

    img_window = np.absolute(filters.prewitt_v(img_window))
    thresh = filters.threshold_otsu(img_window)
    img_window = img_window <= thresh
    labels = measure.label(img_window)

    points = []

    for region in measure.regionprops(labels):
        minr, minc, maxr, maxc = region.bbox
        ratio = float(maxr - minr) / float(maxc - minc)
        heigh = maxr - minr
        area = region.area

        if (ratio > 1 and area > 10 and heigh > 10):
            points.append((minc, minr, maxc, maxr))

    if len(points) > 1:
        points = np.array(points)
        x1 = np.min(points[:, 0])
        x2 = np.max(points[:, 2])

        ransac_model, inliers = measure.ransac(points[:, 0:2], measure.LineModel, 5, 3, max_trials=30)
        points = points[inliers]

        if ransac_model.params[1] != 0:
            average_heigh = int(np.mean(points[:, 3]) - np.mean(points[:, 1]))
            pad_t = average_heigh / 2
            pad_b = average_heigh + (average_heigh / 3)

            y1 = ransac_model.predict_y(x1)
            y2 = ransac_model.predict_y(x2)

            refined_points = [(x1, y1 - pad_t), (x2, y2 - pad_t), (x2, y2 + pad_b), (x1, y1 + pad_b)]
            refined_points = [(x + plate_x, y + plate_y) for (x, y) in refined_points]
            return refined_points

    return plate_points

    # ------------------------------------------------------------
