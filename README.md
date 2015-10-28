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
* shapely==1.5.13
* six==1.9.0
* tornado==4.2

Except the opencv package, all other packages can be installed via pip. Before install the Python packages listed above, please install the geos library by using the follow command:

     sudo apt-get install libgeos-dev libgeos-c1

P.S. This library is required to use the Shapely python package.

### How can I run this software? ###

If you need to build a new model classification, please run the training.py script by using the follow command:

     rm -rf models
     python tranining.py
     mv working models

Now if you want to test a new image using the previously generated model, please run the follow command:

     ./lpdetect filename.jpg

in that filename.jpg is the path to image to be tested.

Our software assumes that there is an annotation file, at the same directory level where lies the filename.jpg. The annotation should be in a file, with same name but extension txt, containing the string "x1,y1,x2,y2,x3,y3,x4,y4,ABC1234", where ABC1234 is the license plate, and x,y are the coordinates in clockwise orientation starting on the top left coordinate (Figure 1).  Assume that the coordinates of the image are on the superior left corner, and that grow to the right and bottow respectively. Images without license plate should contain a string "None".

    ![Annotation](https://github.com/allansp84/license-plate/blob/master/plate_annotation.png "Figure 1 - Orientation of the plate coordinates.")
    Figure 1 - Orientation of the plate coordinates.

### Authors ###

* Allan Pinto
* William Dias
