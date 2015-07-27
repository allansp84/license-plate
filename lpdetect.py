#!/usr/bin/env python

import sys
from datetime import datetime
from modules.protocol import Testing


def time_measurement():
    from modules.datasets import LPRDataset

    dataset = LPRDataset()
    fnames = dataset.metainfo['all_fnames']

    for fname in fnames:
        start = datetime.now()

        Testing(fname).execute()

        print 'Total time elapsed: {0}!'.format((datetime.now() - start))
        sys.stdout.flush()


if __name__ == "__main__":

    Testing(sys.argv[1]).execute()

    # time_measurement()


