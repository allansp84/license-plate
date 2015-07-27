import time
import cv2
from ..utils import *
from ..localization.localization import locatePlate
from ..segmentation import Segmentation
from ..classification import Classification
from ..datasets import LPRDataset

import pdb


class Protocol(object):
    """
    docstring for Protocol
    """

    def __init__(self, args):
        self.data = LPRDataset()
        self.data.dataset_path = args.dataset_path
        self.data.output_path = args.output_path

        self.k_fold = 1
        self.feature = args.feature

        self.n_jobs = int(4*N_JOBS/4)
        self.segmentation_path = 'segmentation'
        self.image = []
        self.points = []

        self.points_fname='{0}/location/points.npy'.format(args.output_path)


    def localization(self):
        start = time.time()

        # -- Loading images
        fnames = self.data.metainfo['all_fnames']

        for fname in fnames:
            img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_COLOR)
            img = cv2.cvtColor(img,cv2.cv.CV_BGR2HSV)[:, :, 2]
            self.points.append(locatePlate(img, fname))
            print fname

        self.points = np.array(self.points)

        try:
            os.makedirs(os.path.dirname(self.points_fname))
        except Exception, e:
            pass

        np.save(self.points_fname, self.points)

        elapsed = (time.time() - start)
        print '\tdone in {0}!'.format(time.strftime("%d days, and %Hh:%Mm:%Ss", time.gmtime(elapsed)))
        sys.stdout.flush()


    def segmentation(self):

        start = time.time()

        all_labels = self.data.metainfo['all_labels']
        all_idxs = self.data.metainfo['all_idxs']

        fnames = self.data.metainfo['all_fnames']

        self.points = np.load(self.points_fname)

        tasks = []
        for idx in xrange(len(fnames)):
            output_path = os.path.join(self.data.output_path, self.segmentation_path)

            tasks += [Segmentation(input_fname=fnames[idx],
                                   points=self.points[idx],
                                   dataset_path=self.data.dataset_path,
                                   output_path=output_path,
                                   feature=self.feature,
                                   persist=True)]

        print "running %d tasks in parallel" % len(tasks)
        RunInParallel(tasks, self.n_jobs).run()

        elapsed = (time.time() - start)
        print '\tdone in {0}!'.format(time.strftime("%d days, and %Hh:%Mm:%Ss", time.gmtime(elapsed)))
        sys.stdout.flush()

    def classification(self):

        start = time.time()

        self.random_state = np.random.RandomState(7)
        for k in xrange(self.k_fold):
            input_path = os.path.join(self.data.output_path, self.segmentation_path)
            metainfo_segmented = self.data.metainfo_images(input_path, ['npy'])

            output_path = os.path.join(input_path, 'svm')
            output_path = output_path.replace('segmentation','classifiers')

            print "output_path",output_path

            clf = Classification(output_path, metainfo_segmented)
            clf.running()

        elapsed = (time.time() - start)
        print '\tdone in {0}!'.format(time.strftime("%d days, and %Hh:%Mm:%Ss", time.gmtime(elapsed)))
        sys.stdout.flush()

        pass

    def execute(self):

        print "localization ..."
        self.localization()

        print "segmentation ..."
        self.segmentation()

        print "building classifiers ..."
        self.classification()
