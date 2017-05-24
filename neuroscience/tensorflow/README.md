# Implementation of MRI benchmark on TensorFlow

The tests were run from main.py. All other files contain code that is specific to a step in the MRI pipeline.<br />
<br />

## Setting up the TensorFlow (v.11) cluster

0.  You may want to use the ami-14cb6c74 (available in W-Oregon) which includes
    a set up distributed TensorFlow 0.11 (bazel) system along with all necessary
    libraries to run the dipy functions as well (you may need to install skimage using pip).

1.  Specify ips of all cluster nodes in a text file (e.g. "ips"). The first ip
    refers to the head node. Make sure that the file ends with exactly one empty
    line.

2.  Run "bazel_dist.sh" on your machine at home with the path to your ip-file
    as commandline argument.

3.  Make sure to also copy your ip-file to your head node since they are needed
    to start working with the cluster (see TfCluster class in distributer.py and
    main.py)
