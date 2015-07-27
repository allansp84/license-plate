import os
import re
import cv2

from datetime import datetime
from itertools import izip

from localization.main import *
from localization.utils import *


# ------------------------------------------------------------

def listFiles(path):
    file_list = []
    regex = re.compile(".*\.jpg")
    for root, sub_folders, files in os.walk(path):
        for f in files:
            if (regex.match(f)):
                file_list.append(f)
    return file_list


# ------------------------------------------------------------

current_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(current_path, "train")
files = sorted(listFiles(dataset_path))

result = []
for i in range(0, 1215):

    # get file path
    file_name = files[i]
    file_path = os.path.join(dataset_path, file_name)
    print '----------------------------------'
    print "treating file {0}".format(file_name)

    # find plate
    start = datetime.now()
    img_full_size = cv2.imread(file_path, 0)
    localization_pred = locatePlate(img_full_size, file_path, plot=False)
    elapsed = (datetime.now() - start)
    print 'Total time elapsed: {0}!'.format(elapsed)

    # verify accuracy
    annotation = np.loadtxt(file_path.replace('.jpg', '.txt'), dtype=str, delimiter=',')
    if annotation.ndim == 1:
        localization_truth = [(int(x), int(y)) for x, y in izip(*[iter(annotation[0:8])] * 2)]
        result.append(localizationAccuracy(localization_truth, localization_pred))
    else:
        result.append(-1)

# show results
histogram = np.histogram(result, bins=10, range=(0, 1))[0]
summation = float(np.sum(histogram))
acc_5 = np.sum(histogram[5: 7])
acc_7 = np.sum(histogram[7:])

print ''
print histogram
print 'total examples = {0}'.format(summation)
print 'acc 0.5 = {0}'.format(float(acc_5) / summation)
print 'acc 0.7 = {0}'.format(float(acc_7) / summation)
print 'acc total = {0}'.format(((float(acc_5) * 0.5) + float(acc_7)) / summation)

# result = np.array(result)
# print np.where((result >= 0.0) & (result < 0.1))[0]
# print np.where((result >= 0.5) & (result < 0.6))[0]
# print np.where((result >= 0.6) & (result < 0.7))[0]
