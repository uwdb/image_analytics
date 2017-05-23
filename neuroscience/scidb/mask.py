import warnings
warnings.filterwarnings("ignore")

from scidbpy import connect
import time
import numpy as np
import dipy.core.gradients as dpg
import nibabel as nib
import subprocess
import os
from os import system

data_loc = "/home/dongfang/download/subjects/101107/"
print "Start calculating mask: ", str(time.ctime())
gtab = dpg.gradient_table(data_loc+'bvals', data_loc+'bvecs', b0_threshold=10)

data_sdb_p0 = sdb.wrap_array("original_half_1st");
data_sdb_p1 = sdb.wrap_array("original_half_2nd");
print "p1 wrapped: ", str(time.ctime())

#there's an I/O error when loading the entire dataset; so I halve it
subset_p0 = data_sdb_p0.compress(sdb.from_array(gtab.b0s_mask[0:144]), axis=3)
print "p0 compressed: ", str(time.ctime())

subset_p1 = data_sdb_p1.compress(sdb.from_array(gtab.b0s_mask[144:288]), axis=3)

subset = sdb.concatenate((subset_p0,subset_p1),axis=3)
print "p0-p1 concated: ", str(time.ctime())

mean_b0 = subset.mean(-1)
print "Mean calculated: ", str(time.ctime())
print "mean_b0.shape = ", mean_b0.shape
sdb.query("store("+mean_b0.name+", mean_b0)")
print "Mean persisted: ", str(time.ctime())
