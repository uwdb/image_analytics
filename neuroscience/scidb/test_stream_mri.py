#!/usr/bin/python

#
#DFZ 11/15/2016: it's hard to control the chunk size read from the 
# stream() interface, see run_mri_stream.output for a concrete idea.
#

import sys
import numpy as np
import dipy.core.gradients as dpg
import os.path as op
from dipy.segment.mask import median_otsu
import nibabel as nib
from dipy.denoise import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from __builtin__ import float
import time

sys.stderr.write("=====> DFZ DEBUG: tests on 1/25/2017.\n")

SUB_ID = 100307
DATA_LOC = "/home/dongfang/download/mri_data/" + str(SUB_ID) +"/"

tm_start = time.time()

#TODO refer to the baseline.py code to generate the mask file

end_of_interaction = 0
while (end_of_interaction != 1):
  header = sys.stdin.readline().rstrip()
  if(header != "0"):
    #We receive a message from the SciDB instance:
    num_lines = int(header)  #how many lines did we get?
    
    sys.stderr.write("=====> DFZ DEBUG 1/25/2017: num_lines = " + str(num_lines) + "\n")

    #Collect all lines into a list:
    input_lines = []
#     list_data = []
#     data = np.zeros((145, 174, 145, 18))
    for i in range(0, num_lines):
      line = sys.stdin.readline().rstrip()
      f = float(line)
      input_lines.append(f)

    #construct the values into a numpy array for MRI
    data = np.reshape(np.asarray(input_lines, dtype=np.float32), (145, 174, 145, 18))
    sys.stderr.write("=====> DFZ DEBUG: data convertion completed.\n")

    #start MRI app

    #masking
    mask = nib.load(DATA_LOC + 'mask.nii.gz').get_data()
    sys.stderr.write("=====> DFZ DEBUG: mask loaded.\n")
 
    #denosing
    sigma = estimate_sigma(data)
    sys.stderr.write("=====> DFZ DEBUG: sigma calculated.\n")
    denoised_data = nlmeans.nlmeans(data, sigma=sigma, mask=mask)
    sys.stderr.write("=====> DFZ DEBUG: denoising completed.\n")
    
    print(2)
    print("Total lines: " + str(num_lines))
    print("Denoising done ----> First line: " + str(input_lines[0]))
    #Print a response: 
#     print(num_lines+1)
#     for i in range(0, num_lines):
#        print("I got\t" + input_lines[i])
#     print("THX!")
    sys.stdout.flush()

    #This will appear in the scidb-sterr.log file:
    sys.stderr.write("I got a chunk with "+ str(num_lines) + " lines of text!\n")
  else:
    #If we receive "0", it means the SciDB instance has no more
    #Data to give us. Here we have the option of also responding with "0"
    #Or sending some other message (i.e. a global sum):
    end_of_interaction = 1
    print("1")
    print("KTHXBYE: subject " + str(SUB_ID) + " done in " + str(time.time() - tm_start) + " seconds.")
    sys.stdout.flush()
    sys.stderr.write("I got the end-of-data message. Exiting.\n")
ok = 0

# So I cannot 'return' or 'print' even after 'return'; the following statements would cause errors
# return 0
# print "Start at " + str(time.ctime())
