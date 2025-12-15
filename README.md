# Confidence Filtering Prostate Pathology Gleason Pattern Classification
Applying confident learning to improve prostate pathology Gleason grading.

   The source code in this project leverages the TensorFlow framework to
build a CNN capable of discriminating Gleason pattern features in small patches
of prostate needle biopsies. The model leverages the VGG-16 network tuned
on ImageNet, but this network can be pruned to fewer than the 5-CNN stages of 
the standard VGG-16 network. Our experiments showed that the 5-stage VGG-16 
was much too powerful and over-trained on our pathology dataset. Ablation tests 
determined that a 2 or 3 stage VGG-16 with a custom classification layer (2-deep FCN)
exhibited better generalization on unseen data.


# Source Code
The source code directory contains a scripts subdirectory with several bash files
used to automate running several experiments (aka tests) and CV-splits, and
a python directory with the TensorFlow training/test code.

## Configuration
Source code is configured via commandline arguments and more permanently via
the dataParameters.py/dataPresentation.py files found in the source/python/configurable
directory (more on those below). Training and testing commandline arguments can all 
be inspected in the arguments.py file. For computing metrics, the parseResultsMetrics.py
file sets up additional arguments specific to metric computation, and model calibration.

### configurable/dataParameters.py
The two source files in the configurable subdirectory are mainly used to set up
paths to source files, and templates for identifying "listing" files for each
experiment (test), split (CV) for training, validation and holdout-test files.

A word about listing files. Instead of forcing files into a single set of
training/validation/holdout subdirectories, the program assumes files are
referenced in "listing" files for any scenario. The listing files are
paths to each file for a particular scenario relative to the path returned
by dataDir() in dataParameters.py.

### configurable/dataPresentation.py
The dataPresentation file may be used to customize how data is presented to
the neural-network. The method names and their arguments are expected by
several main python files (train_multi.py, fine_tune_multi.py, etc.). Some
of these methods are expected to return an iterator (as defined in data.py)
for both training and validation, while some methods only return a single
iterator (namely the testing functions: testSplitNoRANSAC.py and testTrainingForRANSAC.py).


