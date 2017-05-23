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

#SciDB handler
from scidbpy import connect
sdb = connect('http://localhost:8080')


tm_start = time.time()
sys.stderr.write("\n\n=====> DFZ DEBUG 3/10/2017: " + time.ctime() + " OMG I start again! \n")

SUB_ID = 101107
#SUB_ID = 100307
DATA_LOC = "/home/dongfang/mri_data/" + str(SUB_ID) +"/"

denoised_array = np.zeros(1)
pid = os.getpid() #so we need to identify the local array with workerID+processID
denoised_array_pid = "denoised_array_" + str(pid)
denoised_array_pid_flag = denoised_array_pid + "_flag"
sys.stderr.write("=====> 2/26/2017 DFZ DEBUG: denoised_array_pid = " + denoised_array_pid + "\n")

end_of_interaction = 0
while (end_of_interaction != 1):
  header = sys.stdin.readline().rstrip()
  #declare the local denoised array
  if(header != "0"):
    sys.stderr.write("=====> DFZ 2/23/2017: header = " + header + "\n")
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
    data = np.reshape(nparray, (145, 174, 145, 18)) #last param should reflect the chunk size
    sys.stderr.write("=====> DFZ DEBUG: data loading completed.\n")
    
    #masking
    mask = nib.load(DATA_LOC + 'mask.nii.gz').get_data()
    sys.stderr.write("=====> DFZ DEBUG: mask loaded. mask.shape = " + str(mask.shape) + "\n") #full mask shape = ???
 
    #denosing
    sigma = estimate_sigma(data)
    sys.stderr.write("=====> DFZ DEBUG: sigma calculated.\n")
    denoised_data = nlmeans.nlmeans(data, sigma=sigma, mask=mask) #can we ignore the mask?
    sys.stderr.write("=====> 2/25/2017 DFZ DEBUG: denoised_data.shape = " + str(denoised_data.shape) + " \n")

    #denoised_data_py = sdb.from_array(denoised_data)
    #sys.stderr.write("=====> 2/25/2017 DFZ DEBUG: denoised data wrapped\n")

    #step 1: concat denoised_data to denoised_array 

    if not denoised_array_pid_flag in sdb.list_arrays(): #create the init array
      
      #denoised_array_py = sdb.from_array(denoised_data)  
      #sdb.query("create_array(" + denoised_array_pid + ", <f0:float NOT NULL> [i0=0:144,145,0,i1=0:173,174,0,i2=0:144,145,0,i3=0:35,18,0], false)")
      sdb.query("create_array(" + denoised_array_pid_flag + ", <f0:float NOT NULL> [i0=0:0,1,0], true)")
      denoised_array = denoised_data
      sys.stderr.write("=====> 2/25/2017 DFZ DEBUG: after init, denoised_array.shape = " + str(denoised_array.shape) + " \n")
      #sdb.afl.store(denoised_data, denoised_array)
    else:
      sys.stderr.write("=====> 2/25/2017 DFZ DEBUG: before concat, denoised_array.shape = " + str(denoised_array.shape) + " \n")
      denoised_array = np.concatenate((denoised_array, denoised_data), axis=3)
      sys.stderr.write("=====> 2/25/2017 DFZ DEBUG: after concat, denoised_array.shape = " + str(denoised_array.shape) + " \n")


    # if you need interative results:
    print(2)
    print("Total lines: " + str(num_lines))
    print("I'm tired ----> First line: " + str(input_lines[0]))
    sys.stdout.flush()
#This will appear in the scidb-sterr.log file:
    sys.stderr.write(time.ctime() + "I finished a chunk with "+ str(num_lines) + " lines of text!\n")

  else:
    #Persist the local denoised array
    local_array = sdb.from_array(denoised_array, persistent=True)
    sys.stderr.write("=====> 2/26/2017 DFZ DEBUG: local array persisted to " + local_array.name + "\n")
    sdb.query("rename("+local_array.name+","+denoised_array_pid+")")
    sys.stderr.write("=====> 2/26/2017 DFZ DEBUG: local array renamed to " + denoised_array_pid + "\n")

    sys.stderr.write("I got the end-of-data message. Exiting.\n")
    sys.stderr.write("DFZ 2/25/2017: So we are done.\n")

    #If we receive "0", it means the SciDB instance has no more
    #Data to give us. Here we have the option of also responding with "0"
    #Or sending some other message (i.e. a global sum):
    end_of_interaction = 1
    print("1")
#    print("KTHXBYE")
    print("KTHXBYE: subject " + str(SUB_ID) + " done in " + str(time.time() - tm_start) + " seconds; result saved in " + denoised_array_pid)
    sys.stdout.flush()

#ok = 0
# So I cannot 'return' or 'print' even after 'return'; the following statements would cause errors
#exit(0)
# print "Start at " + str(time.ctime())
