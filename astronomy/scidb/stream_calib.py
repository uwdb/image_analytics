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

########################################
####### Setup LSST begins
########################################
visit=289697
ccdnum = 1
HOME_PATH = os.path.expanduser("~")
butler = dafPersist.Butler(HOME_PATH + '/lsst_data/raw')
exposure = butler.get("instcal", visit=visit, ccdnum=ccdnum, filter='g', immediate=True)
mask  = exposure.getMaskedImage().getMask().getArray() #TODO: move this matrix to SciDB
variance = exposure.getMaskedImage().getVariance().getArray() #TODO: move this matrix to SciDB
butler = dafPersist.Butler(HOME_PATH + '/lsst_data/raw')
filename = HOME_PATH + "/lsst_data/raw/crblasted0289697/instcal0289697.1.fits"
fitsHeader = afwImage.readMetadata(filename)
wcs = afwImage.makeWcs(fitsHeader)
charImage = CharacterizeImageTask()
calibrateConfig = CalibrateConfig(doPhotoCal=False, doAstrometry=False)
calibrateTask = CalibrateTask(config=calibrateConfig)
newSkyMapConfig = skymap.discreteSkyMap.DiscreteSkyMapConfig(projection='STG',
                                                             decList=[-4.9325280994132905],
                                                             patchInnerDimensions=[2000, 2000],
                                                             radiusList=[4.488775723429071],
                                                             pixelScale=0.333, rotation=0.0, patchBorder=100,
                                                             raList=[154.10660740464786], tractOverlap=0.0)

hits_skymap = skymap.discreteSkyMap.DiscreteSkyMap(config=newSkyMapConfig)
tract = hits_skymap[0]
sys.stderr.write("=====> DFZ 3/24/2017: mask.shape = " + str(mask.shape) + "\n")
########################################
####### Setup LSST ends
########################################

cnt = 0
end_of_interaction = 0
while (end_of_interaction != 1):
    cnt += 1
    header = sys.stdin.readline().rstrip()
    sys.stderr.write("=====> DFZ 3/24/2017: header = " + header + ", chunk_no = " + str(cnt) + "\n")
    if(header != "0"):
        num_lines = int(header)  #how many lines did we get?
        sys.stderr.write("=====> DFZ 3/24/2017: num_lines = "+str(num_lines)+"\n")

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
        #### LSST Calib ########
        ########################
        #reshape input_lines into a CCD image
        nparray = np.asarray(input_lines, dtype=np.float32)
        data = np.reshape(nparray, (4094, 2046)) #last param should reflect the chunk size
        #process the image
        maskedImage =  bu.makeMaskedImageFromArrays(data, mask, variance)
        image = afwImage.ExposureF(maskedImage)
        image.setWcs(wcs)
        charRes = charImage.characterize(image, exposureIdInfo=None, background=None)
        calibRes = calibrateTask.calibrate(charRes.exposure, exposureIdInfo=None, background=charRes.background, icSourceCat=None)
        #reshape it back to a single string
        data_out = calibRes.exposure.getMaskedImage().getImage().getArray()

        #persist to SciDB
        print(num_lines)
        for i in range(0,4094):
            for j in range(0,2046):
                print(data_out[i,j])
        sys.stdout.flush()

    else:
        sys.stderr.write("=====> DFZ DEBUG: pid " + str(pid) + " finished at " + time.ctime() + "\n" )
        sys.stderr.write("=====> DFZ DEBUG: total run time = " + str(datetime.datetime.now() - tm_start) + " seconds\n")
        print(0)
        #print("pid = " + pid + " finished in " + str(datetime.datetime.now() - tm_start) + " seconds")
        sys.stdout.flush()
        end_of_interaction = 1

