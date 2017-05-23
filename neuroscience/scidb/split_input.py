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

x = np.random.random((5,4))
print x
xsdb = sdb.from_array(x)
#arr = xsdb.toarray()
print xsdb.size
print xsdb.shape
print xsdb.ndim
print xsdb.min(0).toarray()

print "Start loading data from disk: ", str(time.ctime())
img = nib.load('/home/ubuntu/download/data.nii.gz')
data = img.get_data()

print "Start loading data to scidb 1/4: ", str(time.ctime())
data_sdb_0 = sdb.from_array(data[:,:,:,0:72])
#sdb.query("save({A},'mri_0.file',0,'csv')", A=data_sdb_0)
sdb.query("aio_save({A}, 'path=/tmp/mri_0.out', 'instance=0')", A=data_sdb_0)

print "Start loading data to scidb 2/4: ", str(time.ctime())
data_sdb_1 = sdb.from_array(data[:,:,:,72:144])
#sdb.query("save({A},'mri_1.file',1,'csv')", A=data_sdb_1)
sdb.query("aio_save({A}, 'path=/tmp/mri_1.out', 'instance=1')", A=data_sdb_0)

print "Start loading data to scidb 3/4: ", str(time.ctime())
data_sdb_2 = sdb.from_array(data[:,:,:,144:216])
#sdb.query("save({A},'mri_2.file',4294967298,'csv')", A=data_sdb_2)
sdb.query("aio_save({A}, 'path=/tmp/mri_2.out', 'instance=4294967298')", A=data_sdb_0)

print "Start loading data to scidb 4/4: ", str(time.ctime())
data_sdb_3 = sdb.from_array(data[:,:,:,216:288])
#sdb.query("save({A},'mri_3.file',4294967299,'csv')", A=data_sdb_3)
sdb.query("aio_save({A}, 'path=/tmp/mri_3.out', 'instance=4294967299')", A=data_sdb_0)

print "Done: ", str(time.ctime())


