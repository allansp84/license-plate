# License Plate Challenge #

This work refers to final practical project of the Computer vision course offered by Institute of Computing at Unicamp under the responsibility of Prof. Dr. Siome Goldenstein.

### Requirements ###

The packages required to use this software are:

* Opencv 2.4.11
* Numpy 1.9.2
* Pillow==2.9.0
* argparse==1.2.1
* joblib==0.8.4
* matplotlib==1.4.3
* numpy==1.9.2
* pyparsing==2.0.3
* python-dateutil==2.4.2
* scikit-image==0.11.3
* scikit-learn==0.16.1
* scipy==0.15.1
* six==1.9.0
* tornado==4.2

Except the opencv package, all other packages can be installed via pip.

### How can I run this software? ###

If you need to build a new model classification, please run the training.py script by using the follow command:

     rm -rf models
     python tranining.py
     mv working models

Now if you want to test a new image using the previously generated model, please run the follow command:

     ./lpdetect filename

in that filename is the path to image to be tested.

### Authors ###

* Allan Pinto
* William Dias