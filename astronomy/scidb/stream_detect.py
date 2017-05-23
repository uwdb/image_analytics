#/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/bin/python

# Desc:     UDF to calibrate CCDs to patches
# Author:   dzhao@cs.washington.edu
# Date:     3/5/2017

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
from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask, DetectCoaddSourcesConfig
from lsst.pipe.tasks.makeCoaddTempExp import MakeCoaddTempExpTask, MakeCoaddTempExpConfig

#disable the LSST messages
import lsst.log
logger = lsst.log.Log.getDefaultLogger()
logger.setLevel(lsst.log.FATAL)
logger.propagate = False

#DFZ: debug for LSST env variables
pid = str(os.getpid())
#envs = str(os.environ)
#path = str(sys.path)

tm_start = datetime.datetime.now()
sys.stderr.write("\n\n=====> DFZ DEBUG: " + time.ctime() + " OMG I start again! \n")

#DFZ: setup LSST
config = DetectCoaddSourcesConfig()
detectCoaddSources = DetectCoaddSourcesTask(config=config)

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
        #### LSST Detect #######
        ########################
        MAX_X = 2200
        MAX_Y = 2200
        #reshape input_lines into patches across visits
        nparray = np.asarray(input_lines, dtype=np.float32)
        data = np.reshape(nparray, (MAX_X, MAX_Y)) #last param should reflect the chunk size
        #process the patch
        HOME_PATH = os.path.expanduser("~")
        butler = dafPersist.Butler(HOME_PATH + '/lsst_data/raw')
        exposure = butler.get("instcal", visit=289697, ccdnum=1, filter='g', immediate=True)
        mask_old  = exposure.getMaskedImage().getMask().getArray() #TODO: move this matrix to SciDB
        sys.stderr.write("mask_old.shape = " + str(mask_old.shape) + "\n")
        variance_old = exposure.getMaskedImage().getVariance().getArray() #TODO: move this matrix to SciDB
        mask = np.reshape(np.reshape(mask_old,(-1,1))[0:2200*2200], (2200,2200))
        variance = np.reshape(np.reshape(variance_old,(-1,1))[0:2200*2200], (2200,2200))
        sys.stderr.write("mask[0,0] = " + str(mask[0,0]) + ", mask.dtype.type = " + str(mask.dtype.type) + "\n")
        sys.stderr.write("variance[0,0] = " + str(variance[0,0]) + ", variance.dtype.type = " + str(variance.dtype.type) + "\n")

#        mask = np.ones((MAX_X,MAX_Y), dtype=np.uint16)
#        variance = np.ones((MAX_X,MAX_Y), dtype=np.float32)

        maskedImage =  bu.makeMaskedImageFromArrays(data, mask, variance)
        image = afwImage.ExposureF(maskedImage)
        makeCTEConfig = MakeCoaddTempExpConfig()
        makeCTE = MakeCoaddTempExpTask(config=makeCTEConfig)
        modelPsf = makeCTEConfig.modelPsf.apply() 
        image.setPsf(modelPsf)
        detRes = detectCoaddSources.runDetection(image, idFactory=None)

        #persist to SciDB
        print(0)
        sys.stdout.flush()

    else:
        sys.stderr.write("=====> DFZ DEBUG: pid " + str(pid) + " finished at " + time.ctime() + "\n" )
        sys.stderr.write("=====> DFZ DEBUG: total run time = " + str(datetime.datetime.now() - tm_start) + " seconds\n")
        print(0)
        #print("pid = " + pid + " finished in " + str(datetime.datetime.now() - tm_start) + " seconds")
        sys.stdout.flush()
        end_of_interaction = 1

