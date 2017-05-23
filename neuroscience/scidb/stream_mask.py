#!/usr/bin/python

#
#DFZ 11/15/2016: it's hard to control the chunk size read from the
# stream() interface, see run_mri_stream.output for a concrete idea.
#

#the following import block is for testing only
import dipy.core.gradients as dpg
import os.path as op
from dipy.segment.mask import median_otsu
import nibabel as nib
from dipy.denoise import nlmeans
import dipy.core.gradients as dpg
from dipy.denoise.noise_estimate import estimate_sigma
import time
import sys
import numpy as np
import os
from __builtin__ import float

#SciDB handler
#from scidbpy import connect
#sdb = connect('http://localhost:8080')


tm_start = time.time()
sys.stderr.write("\n\n=====> DFZ DEBUG 3/2/2017: " + time.ctime() + " OMG I start again! \n")

SUB_ID = 101107

end_of_interaction = 0
while (end_of_interaction != 1):
  header = sys.stdin.readline().rstrip()
  #declare the local denoised array
  if(header != "0"):
    sys.stderr.write("=====> DFZ 2/24/2017: header = " + header + "\n")
    #We receive a message from the SciDB instance:
    num_lines = int(header)  #how many lines did we get?
    sys.stderr.write("=====> DFZ 1/25/2017: num_lines = "+str(num_lines)+"\n")
    #n_vol = num_lines / 145 / 174 / 145
    #sys.stderr.write("=====> DFZ 1/25/2017: n_vol = "+str(n_vol)+"(should equal to 288/4 )\n")

    #Collect all lines into a list:
    input_lines = []
    for i in range(0, num_lines):
      line = sys.stdin.readline().rstrip()
      try:
        f = float(line)
      except:
        f = 0.0
      input_lines.append(f)

#################################################
############## MRI Logic ########################
#################################################

    #construct the values into a numpy array for MRI
    nparray = np.asarray(input_lines, dtype=np.float32)
#      sys.stderr.write("=====> DFZ DEBUG: convertion completed.\n")
    sys.stderr.write("=====> DFZ DEBUG 2/16/2017: nparray.shape = " + str(nparray.size) + "; len(input_lines) = " + str(len(input_lines)) +"\n")
    mean_b0 = np.reshape(nparray, (145, 174, 145)) #last param should reflect the chunk size
    sys.stderr.write("=====> DFZ DEBUG: data loading completed.\n")

    #masking
    DATA_LOC = "/home/ubuntu/mri_data/101107/"
    gtab = dpg.gradient_table(DATA_LOC + 'bvals', DATA_LOC + 'bvecs', b0_threshold=10)
    mask = median_otsu(mean_b0, 4, 2, False, vol_idx=np.where(gtab.b0s_mask), dilate=1)
    sys.stderr.write("mask: \n")
    sys.stderr.write(str(mask)) #TODO: write it back to SciDB


    # if you need interative results:
    print(2)
    print("Total lines: " + str(num_lines))
    print("I'm tired ----> First line: " + str(input_lines[0]))
    sys.stdout.flush()
#This will appear in the scidb-sterr.log file:
    sys.stderr.write(time.ctime() + "I finished a chunk with "+ str(num_lines) + " lines of text!\n")

  else:

    #If we receive "0", it means the SciDB instance has no more
    #Data to give us. Here we have the option of also responding with "0"
    #Or sending some other message (i.e. a global sum):
    end_of_interaction = 1
    print("1")
#    print("KTHXBYE")
    print("KTHXBYE: subject " + str(SUB_ID) + " done in " + str(time.time() - tm_start) + " seconds")
    sys.stdout.flush()

#ok = 0
# So I cannot 'return' or 'print' even after 'return'; the following statements would cause errors
#exit(0)
# print "Start at " + str(time.ctime())
