#/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/bin/python

# Desc:     UDF to convert calib exps to patches
# Author:   dzhao@cs.washington.edu
# Date:     3/24/2017
# Note:     This cannot be done in parallel within a visit because SciDB stream() processes chunks exclusively

import sys
import os
import time
import datetime
import numpy as np
import pandas as pd

#LSST modules
import lsst.afw.image as afwImage
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
import lsst.afw.image.basicUtils as bu
import lsst.skymap as skymap
import lsst.daf.persistence as dafPersist
import lsst.daf.persistence as dafPersist
from lsst.pipe.tasks.makeCoaddTempExp import MakeCoaddTempExpTask, MakeCoaddTempExpConfig
import lsst.coadd.utils as coaddUtils
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms import CoaddPsf


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
#exposure = butler.get("instcal", visit=visit, ccdnum=ccdnum, filter='g', immediate=True)
#mask  = exposure.getMaskedImage().getMask().getArray() #TODO: move this matrix to SciDB
#variance = exposure.getMaskedImage().getVariance().getArray() #TODO: move this matrix to SciDB
#filename = HOME_PATH + "/lsst_data/raw/crblasted0289697/instcal0289697.1.fits"
#fitsHeader = afwImage.readMetadata(filename)
#wcs = afwImage.makeWcs(fitsHeader)
charImage = CharacterizeImageTask()
calibrateConfig = CalibrateConfig(doPhotoCal=False, doAstrometry=False)
calibrateTask = CalibrateTask(config=calibrateConfig)
makeCTEConfig = MakeCoaddTempExpConfig()
makeCTE = MakeCoaddTempExpTask(config=makeCTEConfig)
newSkyMapConfig = skymap.discreteSkyMap.DiscreteSkyMapConfig(projection='STG',
                                                             decList=[-4.9325280994132905],
                                                             patchInnerDimensions=[2000, 2000],
                                                             radiusList=[4.488775723429071],
                                                             pixelScale=0.333, rotation=0.0, patchBorder=100,
                                                             raList=[154.10660740464786], tractOverlap=0.0)

hits_skymap = skymap.discreteSkyMap.DiscreteSkyMap(config=newSkyMapConfig)
tract = hits_skymap[0]
#sys.stderr.write("=====> DFZ 3/24/2017: mask.shape = " + str(mask.shape) + "\n")

def getSkyInfo(skyMap, xIndex, yIndex, tractId=0):
    tractInfo = skyMap[tractId]
    # patch format is "xIndex,yIndex"
    patchInfo = tractInfo.getPatchInfo((xIndex, yIndex))
    return pipeBase.Struct(
        skyMap = skyMap,
        tractInfo = tractInfo,
        patchInfo = patchInfo,
        wcs = tractInfo.getWcs(),
        bbox = patchInfo.getOuterBBox(),
    )

########################################
####### Setup LSST ends
########################################

end_of_interaction = 0
while (end_of_interaction != 1):
    header = sys.stdin.readline().rstrip()
    sys.stderr.write("=====> DFZ 3/5/2017: header = " + header + "\n")
    if(header != "0"):
        num_lines = int(header)  #how many lines did we get?
        sys.stderr.write("=====> DFZ 3/5/2017: num_lines = "+str(num_lines)+"\n")

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
        #### LSST Patch ########
        ########################
        NUM_CCD = 60
        visit = "0289697" #DFZ: dummy visit number
        v = int(visit) 
        #reshape input_lines into a [ccd, x, y]
        nparray = np.asarray(input_lines, dtype=np.float32)
        data = np.reshape(nparray, (NUM_CCD, 4094, 2046)) #first param should reflect the chunk size
        #process each cal exps 
        ccdsPerPatch =[]
        for ccd_id in range(0, NUM_CCD):
            ccd = ccd_id + 1
            exposure = butler.get("instcal", visit=v, ccdnum=ccd, filter='g', immediate=True)
            mask  = exposure.getMaskedImage().getMask().getArray() #TODO: move this matrix to SciDB
            variance = exposure.getMaskedImage().getVariance().getArray() #TODO: move this matrix to SciDB
            filename = HOME_PATH + "/lsst_data/raw/crblasted0289697/instcal0289697."+str(ccd)+".fits"
            fitsHeader = afwImage.readMetadata(filename)
            wcs = afwImage.makeWcs(fitsHeader)
            calexp_array = data[ccd_id]
            maskedImage =  bu.makeMaskedImageFromArrays(calexp_array, mask, variance)
            image = afwImage.ExposureF(maskedImage)
            image.setWcs(wcs)
            bbox = image.getBBox()
            wcs = image.getWcs()
            corners = bbox.getCorners()
            xIndexMax, yIndexMax = tract.findPatch(wcs.pixelToSky(corners[0][0], corners[0][1])).getIndex()
            xIndexMin, yIndexMin = tract.findPatch(wcs.pixelToSky(corners[2][0], corners[2][1])).getIndex()
            yy = range(yIndexMin, yIndexMax+1)
            xx = range(xIndexMin, xIndexMax+1)
            for yIdx in yy:
                for xIdx in xx:
                    ccdsPerPatch.append((ccd,(xIdx,yIdx)))
                    #sys.stderr.write(str(ccd)+": " +str(xIdx) +", "+ str(yIdx) + "\n")
        sys.stderr.write("length of ccds per patch list: "+str(len(ccdsPerPatch)) + "\n")
        #coadd group-by patch?
        df = pd.DataFrame(ccdsPerPatch)
        dfgby  = df.groupby(1)
        makeCTEConfig = MakeCoaddTempExpConfig()
        makeCTE = MakeCoaddTempExpTask(config=makeCTEConfig)
        coaddTempExpDict = {}
        for a in dfgby.indices:
            coaddTempExpDict[a]={}
        data_out = {}
        for a in dfgby.indices:
            xInd = a[0]
            yInd = a[1]
            skyInfo = getSkyInfo(hits_skymap,xInd,yInd)
            coaddTempExp = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
            coaddTempExp.getMaskedImage().set(np.nan, afwImage.MaskU.getPlaneBitMask("NO_DATA"), np.inf)
            totGoodPix = 0
            didSetMetadata = False
            modelPsf = makeCTEConfig.modelPsf.apply() if makeCTEConfig.doPsfMatch else None
            setInputRecorder=False

            for b in dfgby.get_group(a)[0].ravel():
                sys.stderr.write("group,patch id : "+ str(a) + "\n")
                if not setInputRecorder :
                    ccdsinPatch = len(dfgby.get_group(a)[0].ravel())
                    sys.stderr.write("ccds in patch: "+ str(ccdsinPatch) + "\n")
                    inputRecorder =  makeCTE.inputRecorder.makeCoaddTempExpRecorder(v, ccdsinPatch)
                    setInputRecorder=True
                numGoodPix = 0
                ccd = b
                sys.stderr.write("ccd being added: " + str(ccd) + "; visit " + visit + "\n")
                exposure = butler.get("instcal", visit=v, ccdnum=ccd, filter='g', immediate=True)
                mask  = exposure.getMaskedImage().getMask().getArray() #TODO: move this matrix to SciDB
                variance = exposure.getMaskedImage().getVariance().getArray() #TODO: move this matrix to SciDB
                filename = HOME_PATH + "/lsst_data/raw/crblasted0289697/instcal0289697."+str(ccd)+".fits"
                fitsHeader = afwImage.readMetadata(filename)
                wcs = afwImage.makeWcs(fitsHeader)
                calexp_array = data[ccd-1]
                maskedImage =  bu.makeMaskedImageFromArrays(calexp_array, mask, variance)
                calExp = afwImage.ExposureF(maskedImage)
                calExp.setWcs(wcs)
                ccdId = calExp.getId()
                warpedCcdExp = makeCTE.warpAndPsfMatch.run(calExp, modelPsf=modelPsf,
                                                           wcs=skyInfo.wcs,maxBBox=skyInfo.bbox).exposure
                #if didSetMetadata:
                #    mimg = calExp.getMaskedImage()
                #    mimg *= coaddTempExp.getCalib().getFluxMag0()[0] / (0.001 + calExp.getCalib().getFluxMag0()[0])
                #    del mimg

                numGoodPix = coaddUtils.copyGoodPixels(coaddTempExp.getMaskedImage(),
                                           warpedCcdExp.getMaskedImage(),
                                           makeCTE.getBadPixelMask())
                totGoodPix += numGoodPix
                if numGoodPix > 0 and not didSetMetadata:
                    coaddTempExp.setCalib(warpedCcdExp.getCalib())
                    coaddTempExp.setFilter(warpedCcdExp.getFilter())
                    didSetMetadata = True

                inputRecorder.addCalExp(calExp, ccdId, numGoodPix)

            ##### End loop over ccds here:
            #inputRecorder.finish(coaddTempExp, totGoodPix)
            #if totGoodPix > 0 and didSetMetadata:
            #    coaddTempExp.setPsf(modelPsf if makeCTEConfig.doPsfMatch else
            #        CoaddPsf(inputRecorder.coaddInputs.ccds, skyInfo.wcs))

            #DFZ: so coaddTempExp should be the output?
            sys.stderr.write("(" + str(xInd) + "," + str(yInd) + "): ")
            data_out[(xInd, yInd)] = coaddTempExp.getMaskedImage().getImage().getArray()
            sys.stderr.write("data_out.shape = " + str(data_out[(xInd,yInd)].shape) + "; data_out[500,500] = " + str(data_out[(xInd,yInd)][500,500]) + "\n")

        #persist to SciDB
        sys.stderr.write("=====> DFZ DEBUG: processing time = " + str(datetime.datetime.now() - tm_start) + " seconds\n")
        sys.stderr.write("len(data_out) = " + str(len(data_out)) + "\n")
        tot_patch = len(data_out)
        num_batch = 4
        print(tot_patch/num_batch*2200*2200)
        cnt = 0
        for patch_xy, patch_val in data_out.iteritems():
            if cnt >= tot_patch/num_batch:
                break;
            cnt += 1
            sys.stderr.write("Writing patch #" + str(cnt) + " / " + str(tot_patch) + "\n")
            for x in range(0, 2200):
                for y in range(0, 2200):
                    #print(visit + ' ' + str(patch_xy[0]) + ' ' + str(patch_xy[1]) + ' ' + str(patch_val[x,y]))
                    print(patch_val[x,y])
        sys.stdout.flush()

    else:
        sys.stderr.write("=====> DFZ DEBUG: pid " + str(pid) + " finished at " + time.ctime() + "\n" )
        sys.stderr.write("=====> DFZ DEBUG: total run time = " + str(datetime.datetime.now() - tm_start) + " seconds\n")
        print(0)
        #print("pid = " + pid + " finished in " + str(datetime.datetime.now() - tm_start) + " seconds")
        sys.stdout.flush()
        end_of_interaction = 1

