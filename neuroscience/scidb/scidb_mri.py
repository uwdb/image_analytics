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
import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu

from scidbpy import connect
from scidbpy import save

sdb = connect('http://localhost:8080')

print "Start loading data from disk: ", str(time.ctime())
img = nib.load('/home/ubuntu/download/data.nii.gz')
data = img.get_data()

print "Start loading data to scidb: ", str(time.ctime())
data_sdb_p0 = sdb.from_array(data[:,:,:,0:144])
#sdb.query("save({A},'mri_0.file',0,'csv')", A=data_sdb_0)
#sdb.query("({A}, 'path=/tmp/mri_0.out', 'instance=0')", A=data_sdb_0)
data_sdb_p1 = sdb.from_array(data[:,:,:,144:288])

print "Ingestion done: ", str(time.ctime())

print "data_sdb_1.size = ", data_sdb_p1.size 
print "data_sdb_1.shape = ", data_sdb_p1.shape

print "Start calculating mask: ", str(time.ctime())
gtab = dpg.gradient_table('/home/ubuntu/download/bvals', '/home/ubuntu/download/bvecs', b0_threshold=10)

#there's an I/O error when loading the entire dataset; so I halve it
subset_p0 = data_sdb_p0.compress(sdb.from_array(gtab.b0s_mask[0:144]), axis=3)
subset_p1 = data_sdb_p1.compress(sdb.from_array(gtab.b0s_mask[144:288]), axis=3)
subset = sdb.concatenate((subset_p0,subset_p1),axis=3)
mean_b0 = subset.mean(-1)
print "Mean done: ", str(time.ctime())

#if you want to keep using SciDB for denoising, it means:
#    (1) re-implement dipy.denoise.noise_estimate as a C++ plugin for SciDB
#    (2) re-implement dipy.denoise.nlmeans as a C++ plugin for SciDB
#or, ugly convert it back to numpy array...:
mean_b0_np = mean_b0.toarray()
print "Converting to numpy done: ", str(time.ctime())
print "mean_b0_np.shape = ", mean_b0_np.shape

_, mask = median_otsu(mean_b0_np, 4, 2, False, vol_idx=np.where(gtab.b0s_mask), dilate=1)
print "Mask done: ", str(time.ctime())

