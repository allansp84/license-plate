#!/usr/bin/env python

import sys
import time
import argparse
from modules.protocol import Protocol


def main():

    parser = argparse.ArgumentParser(version='1.0')
    parser.add_argument("--dataset_path", type=str, metavar="str", default='./dataset', help="<dataset_path>")
    parser.add_argument("--output_path", type=str, metavar="str", default='./working', help="<output_path>")
    parser.add_argument("--feature", type=str, metavar="str", default='hog', help="Feature descriptor")
    args = parser.parse_args()

    print "Running Protocol"
    control = Protocol(args)
    control.execute()

if __name__ == "__main__":
    start = time.time()

    main()

    elapsed = (time.time() - start)
    print 'Total time elaposed: {0}!'.format(time.strftime("%d days, and %Hh:%Mm:%Ss", time.gmtime(elapsed)))
    sys.stdout.flush()
