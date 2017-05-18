#LSST use case on pyspark.
def rekey(x):
    import cPickle
    visit = x[0].split("/")[3]
    ccd =  x[0].split("/")[4].split(".")[1]
    return((visit,ccd),x[1])

def unpick(x):
    import cPickle
    return cPickle.loads(x)

# Step1:process ccd.
def processCCDs(x):
    from lsst.ip.isr import IsrTask
    import lsst.pex.config as pexConfig
    import lsst.pipe.base as pipeBase
    from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
    from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
    calibRes = None
    # init Characterize task
    charImageConfig = CharacterizeImageConfig()
    charImage = CharacterizeImageTask()
    #charecterize image
    try:
        charRes = charImage.characterize(x[1])
        calibrateConfig = CalibrateConfig(doPhotoCal=False, doAstrometry=False)
        calibrateTask = CalibrateTask(config=calibrateConfig)
        #caliberate image
        calibRes = calibrateTask.calibrate(charRes.exposure, exposureIdInfo=None, background=charRes.background, icSourceCat=None)
    except Exception as e:
        print "failed to caliberate the image"
        print str(e)
    return(x[0],calibRes)

#2b. flatmap each cdd to multiple patches n:m( 60:116)
def rekeywpatchid(x, bcast_skymap):
    patchList =[]
    if x[1] is not None:
        bbox = x[1].exposure.getBBox()
        wcs = x[1].exposure.getWcs()
        corners = bbox.getCorners()
        xIndexMax, yIndexMax = bcast_skymap.value[0].findPatch(wcs.pixelToSky(corners[0][0], corners[0][1])).getIndex()
        xIndexMin, yIndexMin = bcast_skymap.value[0].findPatch(wcs.pixelToSky(corners[2][0], corners[2][1])).getIndex()
        yy = range(yIndexMin, yIndexMax+1)
        xx = range(xIndexMin, xIndexMax+1)
        for yIdx in yy:
            for xIdx in xx:
                patchList.append(((x[0][0],x[0][1],xIdx,yIdx), x[1]))
        print len(patchList)
    else:
        print "x1 is none!"
        patchList.append(((0,0,0,0),None))
    return patchList

# Step2c.group all ccds belonging to a patch to create patch.
def createPatch(x,bcast_skymap):
    #initialize patch and visit info
    tractInfo = bcast_skymap.value[0]
    visit = int(x[0][0])
    xInd = x[0][1]
    yInd = x[0][2]
    retval = ((visit,xInd,yInd,0),None)
    if (xInd !=0 and yInd !=0 and visit !=0):
        patchInfo = tractInfo.getPatchInfo((xInd, yInd))
        skyInfo  = pipeBase.Struct(
            skyMap =  bcast_skymap.value,
            tractInfo = tractInfo,
            patchInfo = patchInfo,
            wcs = tractInfo.getWcs(),
            bbox = patchInfo.getOuterBBox(),
        )
        #create empty tempexp
        coaddTempExp = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddTempExp.getMaskedImage().set(numpy.nan, afwImage.MaskU.getPlaneBitMask("NO_DATA"), numpy.inf)
        totGoodPix = 0
        didSetMetadata = False
        #configure the task
        makeCTEConfig = MakeCoaddTempExpConfig()
        makeCTE = MakeCoaddTempExpTask(config=makeCTEConfig)
        modelPsf = makeCTEConfig.modelPsf.apply() if makeCTEConfig.doPsfMatch else None
        #number of ccds in the patch is the number in the group
        ccdsInPatch = len(x[1])
        inputRecorder = makeCTE.inputRecorder.makeCoaddTempExpRecorder(visit, ccdsInPatch)
        for ccd in x[1]:
            calExp = ccd[1].exposure
            ccdId = calExp.getId()
            if didSetMetadata: #then scale image to zeropoint of first ccd
                mimg = calExp.getMaskedImage()
                mimg *= (coaddTempExp.getCalib().getFluxMag0()[0] / calExp.getCalib().getFluxMag0()[0])
                del mimg
            warpedCcdExp = makeCTE.warpAndPsfMatch.run(calExp, modelPsf=modelPsf,
                wcs=skyInfo.wcs,maxBBox=skyInfo.bbox).exposure
            numGoodPix = coaddUtils.copyGoodPixels(coaddTempExp.getMaskedImage(),
                                                 warpedCcdExp.getMaskedImage(),
                                                 makeCTE.getBadPixelMask())
            totGoodPix += numGoodPix
            print ccdId, numGoodPix
            if numGoodPix > 0 and not didSetMetadata:
                coaddTempExp.setCalib(warpedCcdExp.getCalib())
                coaddTempExp.setFilter(warpedCcdExp.getFilter())
                didSetMetadata = True
            inputRecorder.addCalExp(calExp, ccdId, numGoodPix)
        inputRecorder.finish(coaddTempExp, totGoodPix)
        if totGoodPix > 0 and didSetMetadata:
            coaddTempExp.setPsf(modelPsf if makeCTEConfig.doPsfMatch else CoaddPsf(inputRecorder.coaddInputs.ccds, skyInfo.wcs))
        retval = ((visit,xInd,yInd,ccdsInPatch),coaddTempExp)
    return retval

#helper function for merge co-adds
def prepInput(assembleTask,CTEs):
    from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask, AssembleCoaddConfig
    import lsst.afw.math as afwMath
    statsCtrl = afwMath.StatisticsControl()
    statsCtrl.setNumSigmaClip(assembleTask.config.sigmaClip)
    statsCtrl.setNumIter(assembleTask.config.clipIter)
    statsCtrl.setAndMask(assembleTask.getBadPixelMask())
    statsCtrl.setNanSafe(True)
    newDataIdList = [] #make clean list incase scaling failed. output lists should all be same length
    weightList = []
    imageScalerList = []
    tempExpName = "Deep"
    for CTE in CTEs:
        dataId = CTE[0][0]
        tempExp = CTE[1]
        maskedImage = tempExp.getMaskedImage()
        imageScaler = assembleTask.scaleZeroPoint.computeImageScaler(
            exposure = tempExp,
            dataRef = None,
        )
        statObj = afwMath.makeStatistics(maskedImage.getVariance(), maskedImage.getMask(),
            afwMath.MEANCLIP, statsCtrl)
        meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP);
        weight = 1.0 / float(meanVar)
        if not numpy.isfinite(weight):
            print("Non-finite weight for %s: skipping" % (dataId))
            continue
        print("Weight of %s %s = %0.3f" % (tempExpName, dataId, weight))
        del maskedImage
        newDataIdList.append(dataId)
        weightList.append(weight)
        imageScalerList.append(imageScaler)
        del tempExp
    return pipeBase.Struct(dataIdList=newDataIdList, weightList=weightList,
                           imageScalerList=imageScalerList)

#Step3: merge co-adds to create merge coadds
def mergeCoadd(x):
    from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask, AssembleCoaddConfig
    import lsst.afw.math as afwMath
    import lsst.meas.algorithms as measAlg
    #get sky info
    tractInfo = bcast_skymap.value[0]
    xInd = x[0][0]
    yInd = x[0][1]
    retval = (x[0],None)
    if (xInd!=0 and yInd!=0):
        patchInfo = tractInfo.getPatchInfo((xInd, yInd))
        skyInfo  = pipeBase.Struct(
            skyMap =  bcast_skymap.value,
            tractInfo = tractInfo,
            patchInfo = patchInfo,
            wcs = tractInfo.getWcs(),
            bbox = patchInfo.getOuterBBox(),
        )
        #initialize task
        config = AssembleCoaddConfig()
        assembleTask = AssembleCoaddTask(config=config)
        #prep Inp
        imageScalerRes = prepInput(assembleTask,x[1])
        mask = assembleTask.getBadPixelMask()
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(assembleTask.config.sigmaClip)
        statsCtrl.setNumIter(assembleTask.config.clipIter)
        statsCtrl.setAndMask(mask)
        statsCtrl.setNanSafe(True)
        statsCtrl.setWeighted(True)
        statsCtrl.setCalcErrorFromInputVariance(True)
        for plane, threshold in assembleTask.config.maskPropagationThresholds.items():
            bit = afwImage.MaskU.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)
        statsFlags = afwMath.MEAN
        #create empty coaddExposures
        coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddExposure.setCalib(assembleTask.scaleZeroPoint.getCalib())
        coaddExposure.getInfo().setCoaddInputs(assembleTask.inputRecorder.makeCoaddInputs())
        #most important thing is the psf
        idx = 0
        for CTE in x[1]:
            ccdsinPatch = CTE[0][3]
            print "CCDs in patch: "+ str(ccdsinPatch)
            tempExp = CTE[1]
            weight = imageScalerRes.weightList[idx]
            idx += 1
            coaddExposure.setFilter(tempExp.getFilter())
            coaddInputs = coaddExposure.getInfo().getCoaddInputs()
            assembleTask.inputRecorder.addVisitToCoadd(coaddInputs, tempExp, weight)
        coaddInputs.ccds.reserve(ccdsinPatch)
        coaddInputs.visits.reserve(len(imageScalerRes.dataIdList))
        psf = measAlg.CoaddPsf(coaddInputs.ccds, coaddExposure.getWcs())
        coaddExposure.setPsf(psf)
        maskedImageList = afwImage.vectorMaskedImageF()
        coaddMaskedImage = coaddExposure.getMaskedImage()
        for dataId, imageScaler, exposure in zip(imageScalerRes.dataIdList,
                                                 imageScalerRes.imageScalerList,
                                                 x[1]):
            exp = exposure[1]
            print dataId, imageScaler, exposure
            maskedImage = exp.getMaskedImage()
            imageScaler.scaleMaskedImage(maskedImage)
            maskedImageList.append(maskedImage)
        maskedImage = afwMath.statisticsStack(maskedImageList,
                                              statsFlags, statsCtrl,
                                              imageScalerRes.weightList)
        coaddMaskedImage.assign(maskedImage, skyInfo.bbox)
        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(),
                                    coaddMaskedImage.getVariance())
        retval = (x[0], coaddExposure)
    return retval

#step4: detect sources.
from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask, DetectCoaddSourcesConfig
def detectSources(coaddExposure):
    config = DetectCoaddSourcesConfig()
    detectCoaddSources = DetectCoaddSourcesTask(config=config)
    retval = None
    if coaddExposure is not None:
        retval = detectCoaddSources.runDetection(coaddExposure, idFactory=None)
    return retval

def getSourceCount(x):
    if x[1] is not None:
        return((x[0][0],x[0][1]),len(x[1]))
    else:
        return((x[0][0],x[0][1]),0)

from pyspark import SparkContext
from pyspark import SparkConf
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.processCcd import ProcessCcdTask
from lsst.obs.decam.decamNullIsr import DecamNullIsrTask
from lsst.ip.isr import IsrTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
import lsst.skymap as skymap
from lsst.pipe.tasks.makeCoaddTempExp import MakeCoaddTempExpTask, MakeCoaddTempExpConfig
import lsst.afw.image as afwImage
import lsst.coadd.utils as coaddUtils
from lsst.meas.algorithms import CoaddPsf
import sys
import datetime
import numpy

from pyspark import SparkContext
sc = SparkContext()
if len(sys.argv) == 1:
    print("Must pass in the number of visits")
    sys.exit(0)

numVisits = int(sys.argv[1])

visits = [ "..."][:numVisits]

s3Paths = ["..."+str(visit) for visit in visits]
for path in s3Paths:
    print path

imgRDDs = [sc.binaryFiles(s3path, minPartitions = 60) for s3path in s3Paths]
imgRDD = sc.union(imgRDDs).map(rekey).mapValues(unpick).cache()
imgRDD.count()

#Step1: process ccds
calExpRDD = imgRDD.map(processCCDs)


#2a. Initialize skymap
newSkyMapConfig  = skymap.discreteSkyMap.DiscreteSkyMapConfig(projection='STG',
                                                              decList=[-4.9325280994132905],
                                                               patchInnerDimensions=[2000, 2000],
                                                              radiusList=[4.488775723429071],
                                                              pixelScale=0.333, rotation=0.0, patchBorder=100,
                                                              raList=[154.10660740464786], tractOverlap=0.0)

hits_skymap = skymap.discreteSkyMap.DiscreteSkyMap(config = newSkyMapConfig)
# send the skymapout as a broadcast variable
bcast_skymap = sc.broadcast(hits_skymap)
#Step2.a flatmap ccds to patches
calExpRDD2 = calExpRDD.flatMap(lambda x: rekeywpatchid(x, bcast_skymap))
#Step2.b group by visit AND patch id (x,y)
calExpRDD3 = calExpRDD2.groupBy(lambda x: (x[0][0],x[0][2],x[0][3]))
#Step2c. assemble patches from each group
coAddTempExpRDD = calExpRDD3.map(lambda x:createPatch(x,bcast_skymap))

#Step3: merge coadds to create merged co-adds.
#Step3a - group by patch id (x,y)
coAddTempExpRDD2 = coAddTempExpRDD.groupBy(lambda x: (x[0][1],x[0][2]))
#Step3b:merge co-add
coaddExposuresRDD = coAddTempExpRDD2.map(mergeCoadd)

#step4: detect sources
sourcesRDD = coaddExposuresRDD.mapValues(detectSources).cache()
sourcesRDD.count()


