
from myria import *
import numpy
from myria.connection import MyriaConnection
from myria.relation import MyriaRelation
from myria.udf import MyriaPythonFunction
from raco.types import STRING_TYPE, BOOLEAN_TYPE, LONG_TYPE, BLOB_TYPE

connection = MyriaConnection(rest_url='http://localhost:8753',execution_url='http://localhost:8080')

def processCCDs(dt):
    from lsst.ip.isr import IsrTask
    import lsst.pex.config as pexConfig
    from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
    from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
    img = dt[0][0]
    calibRes = None

    charImage = CharacterizeImageTask()
    calibrateConfig = CalibrateConfig(doPhotoCal=False, doAstrometry=False)
    calibrateTask = CalibrateTask(config=calibrateConfig)

    try:
        # charecterize image
        charRes = charImage.characterize(img)
        #caliberate image
        calibRes = calibrateTask.calibrate(charRes.exposure, exposureIdInfo=None, background=charRes.background, icSourceCat=None)
    except Exception as e:
        print "failed to caliberate the image"
        print str(e)
    return calibRes

MyriaPythonFunction(processCCDs, BLOB_TYPE).register()


def rekeywpatchid(dt):
    import cPickle
    patchList =[]
    ccdlist = dt[0]

    ccd = ccdlist[0][0]
    skymap = cPickle.load(open("...", 'rb'))
    tract = skymap[0]
    if ccd is not None:
        bbox = ccd.exposure.getBBox()
        wcs = ccd.exposure.getWcs()
        corners = bbox.getCorners()
        xIndexMax, yIndexMax = tract.findPatch(wcs.pixelToSky(corners[0][0], corners[0][1])).getIndex()
        xIndexMin, yIndexMin = tract.findPatch(wcs.pixelToSky(corners[2][0], corners[2][1])).getIndex()
        yy = range(yIndexMin, yIndexMax+1)
        xx = range(xIndexMin, xIndexMax+1)
        for yIdx in yy:
            for xIdx in xx:
                a = xIdx*100+yIdx
                patchList.append(a)
                print a
                print type(a)
        print len(patchList)
    else:
        print "ccd is none!"
        patchList.append(0)
    return patchList

MyriaPythonFunction(rekeywpatchid, BLOB_TYPE, multivalued=True).register()


def createPatch(dt):
    #initialize patch and visit info
    import cPickle
    import numpy
    from lsst.pipe.tasks.makeCoaddTempExp import MakeCoaddTempExpTask, MakeCoaddTempExpConfig
    import lsst.coadd.utils as coaddUtils
    from lsst.meas.algorithms import CoaddPsf
    import lsst.pipe.base as pipeBase
    import lsst.afw.image as afwImg
    skymap = cPickle.load(open("...", 'rb'))
    print ("loaded skymap!")
    tract = skymap[0]
    visit = 0
    tuplist = dt[1]
    print ("length of tuplist..." +str(len(tuplist)))
    retval = None
    patchInfo= None
    ccdsInPatch = len(tuplist)
    skyInfo = None
    #get patch id.
    if(len(tuplist)>0):
        a = tuplist[0][2]
        xInd = a//100
        yInd =a%100
        visit = tuplist[0][3]
        print visit, xInd, yInd, ccdsInPatch
        if (xInd !=0 and yInd !=0 and visit !=0):
            patchInfo = tract.getPatchInfo((xInd, yInd))
            skyInfo  = pipeBase.Struct(
                    skyMap =  skymap,
                    tractInfo = tract,
                    patchInfo = patchInfo,
                    wcs = tract.getWcs(),
                    bbox = patchInfo.getOuterBBox(),
                )
            #create empty tempexp
            coaddTempExp = afwImg.ExposureF(skyInfo.bbox, skyInfo.wcs)
            coaddTempExp.getMaskedImage().set(numpy.nan, afwImg.MaskU.getPlaneBitMask("NO_DATA"), numpy.inf)
            totGoodPix = 0
            didSetMetadata = False
            #configure the task
            makeCTEConfig = MakeCoaddTempExpConfig()
            makeCTE = MakeCoaddTempExpTask(config=makeCTEConfig)
            modelPsf = makeCTEConfig.modelPsf.apply() if makeCTEConfig.doPsfMatch else None
            inputRecorder = makeCTE.inputRecorder.makeCoaddTempExpRecorder(visit, ccdsInPatch)
        for ccd in tuplist:
            calExp = ccd[1].exposure
            ccdId = calExp.getId()
            numGoodPix=0

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

            if numGoodPix > 0 and not didSetMetadata:
                coaddTempExp.setCalib(warpedCcdExp.getCalib())
                coaddTempExp.setFilter(warpedCcdExp.getFilter())
                didSetMetadata = True

            inputRecorder.addCalExp(calExp, ccdId, numGoodPix)
        inputRecorder.finish(coaddTempExp, totGoodPix)
        if totGoodPix > 0 and didSetMetadata:
             coaddTempExp.setPsf(modelPsf if makeCTEConfig.doPsfMatch else CoaddPsf(inputRecorder.coaddInputs.ccds, skyInfo.wcs))
             retval = (coaddTempExp,ccdsInPatch)
    return retval


MyriaPythonFunction(createPatch, BLOB_TYPE).register()


    #helper function for merge co-adds
def prepInput(assembleTask,CTEs):
    from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask, AssembleCoaddConfig
    import lsst.afw.math as afwMath
    import lsst.pipe.base as pipeBase
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
        #print CTE
        dataId = CTE[1]
        tempExp = CTE[3][0]
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
def mergeCoadd(dt):
    from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask, AssembleCoaddConfig
    import lsst.afw.math as afwMath
    import lsst.meas.algorithms as measAlg
    import lsst.afw.image       as afwImg
    import lsst.pipe.base as pipeBase
    import lsst.coadd.utils as coaddUtils
    import cPickle
    #get sky info
    skymap = cPickle.load(open("...", 'rb'))
    tract = skymap[0]
    tuplist = dt[1]
    retval = None
    ccdInVisit = len(tuplist)

    #get patch id.
    if(len(tuplist)>0):
        a = tuplist[0][2]
        xInd = a//100
        yInd =a%100
        if (xInd!=0 and yInd!=0):
            patchInfo = tract.getPatchInfo((xInd, yInd))
            skyInfo  = pipeBase.Struct(
                skyMap =  skymap,
                tractInfo = tract,
                patchInfo = patchInfo,
                wcs = tract.getWcs(),
                bbox = patchInfo.getOuterBBox(),
            )

        #initialize task
        config = AssembleCoaddConfig()
        assembleTask = AssembleCoaddTask(config=config)
        #prep Inp
        imageScalerRes = prepInput(assembleTask,tuplist)
        mask = assembleTask.getBadPixelMask()
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(assembleTask.config.sigmaClip)
        statsCtrl.setNumIter(assembleTask.config.clipIter)
        statsCtrl.setAndMask(mask)
        statsCtrl.setNanSafe(True)
        statsCtrl.setWeighted(True)
        statsCtrl.setCalcErrorFromInputVariance(True)
        for plane, threshold in assembleTask.config.maskPropagationThresholds.items():
            bit = afwImg.MaskU.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)
        statsFlags = afwMath.MEAN
        #create empty coaddExposures
        coaddExposure = afwImg.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddExposure.setCalib(assembleTask.scaleZeroPoint.getCalib())
        coaddExposure.getInfo().setCoaddInputs(assembleTask.inputRecorder.makeCoaddInputs())
        #most important thing is the psf
        idx=0
        for CTE in tuplist:
            tempExp = CTE[3][0]
            ccdsinPatch = CTE[3][1]
            weight = imageScalerRes.weightList[idx]
            idx += 1
            coaddExposure.setFilter(tempExp.getFilter())
            coaddInputs = coaddExposure.getInfo().getCoaddInputs()
            assembleTask.inputRecorder.addVisitToCoadd(coaddInputs, tempExp, weight)
        coaddInputs.ccds.reserve(ccdsinPatch)
        coaddInputs.visits.reserve(len(imageScalerRes.dataIdList))
        psf = measAlg.CoaddPsf(coaddInputs.ccds, coaddExposure.getWcs())
        coaddExposure.setPsf(psf)
        maskedImageList = afwImg.vectorMaskedImageF()
        coaddMaskedImage = coaddExposure.getMaskedImage()
        for dataId, imageScaler, exposure in zip(imageScalerRes.dataIdList,
                                                 imageScalerRes.imageScalerList,
                                                 tuplist):
            exp = exposure[3][0]
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
        retval = coaddExposure
    return retval



MyriaPythonFunction(mergeCoadd, BLOB_TYPE).register()


def detectSources(dt):
    from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask, DetectCoaddSourcesConfig
    config = DetectCoaddSourcesConfig()
    detectCoaddSources = DetectCoaddSourcesTask(config=config)
    retval = None
    imagelist = dt[1]
    print len(imagelist)
    for image in imagelist:
        retval = detectCoaddSources.runDetection(image[0], idFactory=None)
    return retval

MyriaPythonFunction(detectSources, BLOB_TYPE).register()


####create skymap
import lsst.skymap as skymap
import cPickle
newSkyMapConfig  = skymap.discreteSkyMap.DiscreteSkyMapConfig(projection='STG',
                                                                  decList=[-4.9325280994132905],
                                                                   patchInnerDimensions=[2000, 2000],
                                                                  radiusList=[4.488775723429071],
                                                                  pixelScale=0.333, rotation=0.0, patchBorder=100,
                                                                  raList=[154.10660740464786], tractOverlap=0.0)

skymap = skymap.discreteSkyMap.DiscreteSkyMap(config = newSkyMapConfig)
cPickle.dump(skymap, open("skymap.p",'wb'))
