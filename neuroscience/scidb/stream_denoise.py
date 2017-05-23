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
from dipy.denoise.noise_estimate import estimate_sigma
import time
import sys
import numpy as np
import os
from __builtin__ import float


tm_start = time.time()
sys.stderr.write("\n\n=====> DFZ DEBUG: " + time.ctime() + " OMG I start again! \n")

SUB_ID = 101107
DATA_LOC = "/home/ubuntu/mri_data/" + str(SUB_ID) +"/"


end_of_interaction = 0
while (end_of_interaction != 1):
  header = sys.stdin.readline().rstrip()
  #declare the local denoised array
  if(header != "0"):
    sys.stderr.write("=====> DFZ 3/10/2017: header = " + header + "\n")
    #We receive a message from the SciDB instance:
    num_lines = int(header)  #how many lines did we get?
    sys.stderr.write("=====> DFZ 3/10/2017: num_lines = "+str(num_lines)+"\n")
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
    sys.stderr.write("=====> DFZ DEBUG 3/10/2017: nparray.shape = " + str(nparray.size) + "; len(input_lines) = " + str(len(input_lines)) +"\n")
    data = np.reshape(nparray, (145, 174, 145, 9)) #last param should reflect the chunk size
    sys.stderr.write("=====> DFZ DEBUG: data loading completed.\n")
    
    #masking
    mask = nib.load(DATA_LOC + 'mask.nii.gz').get_data()
    sys.stderr.write("=====> DFZ DEBUG: mask loaded. mask.shape = " + str(mask.shape) + "\n") #full mask shape = ???
 
    #denosing
    sigma = estimate_sigma(data)
    sys.stderr.write("=====> DFZ DEBUG: sigma calculated.\n")
    denoised_data = nlmeans.nlmeans(data, sigma=sigma, mask=mask) #can we ignore the mask?
    sys.stderr.write("=====> 2/25/2017 DFZ DEBUG: denoised_data.shape = " + str(denoised_data.shape) + " \n")

    #write it back to scidb
    print(145*174*145*9)
    for i in range(0, 145):
        for j in range(0, 174):
            for k in range(0, 145):
                for l in range(0,9):
                    print(data[i,j,k,l])

    sys.stderr.write(time.ctime() + ": I finished a chunk with "+ str(num_lines) + " lines of text!\n")
    sys.stdout.flush()

  else:
    sys.stderr.write("I got the end-of-data message. Exiting.\n")
    end_of_interaction = 1
    print(0)
    sys.stdout.flush()

#ok = 0
# So I cannot 'return' or 'print' even after 'return'; the following statements would cause errors
#exit(0)
# print "Start at " + str(time.ctime())
