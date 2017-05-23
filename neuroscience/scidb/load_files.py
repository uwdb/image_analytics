#
# Multi-node SciDB implementation of MRI
#
# dzhao@uw.edu, 2016
#

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import nibabel as nib
from scidbpy import connect
from scidbpy import save

sdb = connect('http://localhost:8080')

#sdb.query("remove(mri_4_0)")
#sdb.query("remove(mri_4_3)")

#DFZ: local read, completed in 90 seconds
print "Start loading data from scidb 1/4: ", str(time.ctime())
#sdb.query("create array mri_4_0 <x:double> [i]")
#sdb.query("load(mri_4_0,'mri_0.file', 0, 'csv')")
sdb.query("store(aio_input('paths=/tmp/mri_0.out', 'instances=0', 'num_attributes=1'), mri_4_0)")

#DFZ: remote read, not working
print "Start loading data from scidb 4/4: ", str(time.ctime())
#sdb.query("create array mri_4_3 <x:double> [i]")
sdb.query("store(aio_input('paths=/tmp/mri_3.out', 'instances=4294967299', 'num_attributes=1'), mri_4_3)")

print "Done: ", str(time.ctime())


