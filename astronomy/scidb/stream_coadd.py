#/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/bin/python

# Author:   dzhao@cs.washington.edu

import sys
import os
import time
import datetime
import numpy as np

import lsst.afw.image as afwImage
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
import lsst.afw.image.basicUtils as bu
import lsst.skymap as skymap
import lsst.daf.persistence as dafPersist
import lsst.daf.persistence as dafPersist

#disable the LSST messages
import lsst.log
logger = lsst.log.Log.getDefaultLogger()
logger.setLevel(lsst.log.FATAL)

#DFZ: debug for LSST env variables
pid = str(os.getpid())
#envs = str(os.environ)
#path = str(sys.path)

tm_start = datetime.datetime.now()
sys.stderr.write("\n\n=====> DFZ DEBUG: " + time.ctime() + " OMG I start again! \n")

end_of_interaction = 0
while (end_of_interaction != 1):
    header = sys.stdin.readline().rstrip()
    sys.stderr.write("=====> DFZ: header = " + header + "\n")
    if(header != "0"):
        num_lines = int(header)  #how many lines did we get?
        sys.stderr.write("=====> DFZ: num_lines = "+str(num_lines)+"\n")

        input_lines = []
        output_lines = []
        for i in range(0, num_lines):
            line = sys.stdin.readline().rstrip()
            if not i%1000000:
                sys.stderr.write("line "+str(i)+"/"+str(num_lines)+": " + line + "\n")
            try:
                f = float(line)
            except:
                f = 0.0
            input_lines.append(f)
            
        ########################
        #### LSST Coadd ########
        ########################
        NUM_VISITS = 2
        MAX_X = 2200
        MAX_Y = 2200
        #reshape input_lines into patches across visits
        nparray = np.asarray(input_lines, dtype=np.float32)
        data = np.reshape(nparray, (NUM_VISITS, MAX_X, MAX_Y)) #last param should reflect the chunk size
        #process the patch
        data_coadd = np.sum(data, axis=0)
        cnt = 0
        sys.stderr.write("Starting sigma clipping\n")
        while True:
            sys.stderr.write("Sigma clipping at loop #" + str(cnt) + "\n")
            cnt += 1
            for x in range(0, MAX_X):
                for y in range(0, MAX_Y):
                    mean = np.mean(data[:,x,y])
                    stdv = np.std(data[:,x,y])
                    min = mean - 5 * stdv
                    max = mean + 5 * stdv
                    for v in range(0, NUM_VISITS):
                        if data[v,x,y] < min or data[v,x,y] > max:
                            data[v,x,y] = None
                if not x%100:
                    sys.stderr.write("x = " + str(x) + ", y = " + str(y) + "\n")
            data_coadd_new = np.sum(data, axis=0) 
            if np.array_equal(data_coadd, data_coadd_new):
                break
            else:
                data_coadd = data_coadd_new

        #persist to SciDB
        print(MAX_X * MAX_Y)
        for i in range(0,MAX_X):
            for j in range(0,MAX_Y):
                print(data_coadd[i,j])
        sys.stdout.flush()

    else:
        sys.stderr.write("=====> DFZ DEBUG: pid " + str(pid) + " finished at " + time.ctime() + "\n" )
        sys.stderr.write("=====> DFZ DEBUG: total run time = " + str(datetime.datetime.now() - tm_start) + " seconds\n")
        print(0)
        #print("pid = " + pid + " finished in " + str(datetime.datetime.now() - tm_start) + " seconds")
        sys.stdout.flush()
        end_of_interaction = 1

