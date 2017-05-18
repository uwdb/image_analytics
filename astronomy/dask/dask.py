import boto3
import os.path as op
import sys, time, itertools, timeit
import pandas as pd
import numpy
import datetime
from distributed import Executor
import distributed

# try out one exposure
visits = [ "visitids"]

def getSkyInfo(skyMap, xIndex, yIndex, tractId=0):
    import lsst.pipe.base as pipeBase
    tractInfo = skyMap[tractId]
    # patch format is "xIndex,yIndex"
    patchInfo = tractInfo.getPatchInfo((xIndex, yIndex))
    return pipeBase.Struct(
        skyMap=skyMap,
        tractInfo=tractInfo,
        patchInfo=patchInfo,
        wcs=tractInfo.getWcs(),
        bbox=patchInfo.getOuterBBox(),
    )


def detect(exp):
    from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask, DetectCoaddSourcesConfig
    if exp is None:
        return None
    config = DetectCoaddSourcesConfig()
    detectCoaddSources = DetectCoaddSourcesTask(config=config)
    r = detectCoaddSources.runDetection(exp, idFactory=None)
    return r


def prepareInputs(cteList, dataIdList, assembleTask):
    import lsst.afw.math as afwMath
    import lsst.pipe.base as pipeBase

    """!
    Prepare the input warps for coaddition by measuring the weight for each warp and the scaling
    for the photometric zero point.

    Each coaddTempExp has its own photometric zeropoint and background variance. Before coadding these
    coaddTempExps together, compute a scale factor to normalize the photometric zeropoint and compute the
    weight for each coaddTempExp.

    param[in] refList: List of data references to tempExp
    return Struct:
    - tempExprefList: List of data references to tempExp
    - weightList: List of weightings
    - imageScalerList: List of image scalers
    """
    statsCtrl = afwMath.StatisticsControl()
    statsCtrl.setNumSigmaClip(assembleTask.config.sigmaClip)
    statsCtrl.setNumIter(assembleTask.config.clipIter)
    statsCtrl.setAndMask(assembleTask.getBadPixelMask())
    statsCtrl.setNanSafe(True)
    # compute tempExpRefList: a list of tempExpRef that actually exist
    # and weightList: a list of the weight of the associated coadd tempExp
    # and imageScalerList: a list of scale factors for the associated coadd tempExp
    newDataIdList = []  # make clean list incase scaling failed. output lists should all be same length
    weightList = []
    imageScalerList = []
    tempExpName = "Deep"
    for dataId, tempExp in zip(dataIdList, cteList):
        maskedImage = tempExp.getMaskedImage()
        try:
            imageScaler = assembleTask.scaleZeroPoint.computeImageScaler(
                exposure=tempExp,
                dataRef=None,
            )
        except Exception as e:
            print "failed to scale the image"
            return None
        statObj = afwMath.makeStatistics(maskedImage.getVariance(), maskedImage.getMask(),
                                         afwMath.MEANCLIP, statsCtrl)
        meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP);
        weight = 1.0 / float(meanVar)
        if not numpy.isfinite(weight):
            print("Non-finite weight for %s: skipping" % (dataId))
            continue
        print("Weight of %s %s = %0.3f" % (tempExpName, dataId, weight))
        del maskedImage
        del tempExp
        newDataIdList.append(dataId)
        weightList.append(weight)
        imageScalerList.append(imageScaler)
    return pipeBase.Struct(dataIdList=newDataIdList, weightList=weightList,
                           imageScalerList=imageScalerList)


def mergeCoadd(a, dfgby_raveled, coaddTempExpDict_a, hits_skymap):
    from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask, AssembleCoaddConfig
    import lsst.afw.math as afwMath
    import lsst.afw.image as afwImage
    import lsst.coadd.utils as coaddUtils
    import lsst.meas.algorithms as measAlg

    config = AssembleCoaddConfig()
    assembleTask = AssembleCoaddTask(config=config)
    ccdsinPatch = len(dfgby_raveled)  # len(dfgby.get_group(a)[0].ravel())
    xInd = a[0]
    yInd = a[1]

    print xInd, yInd
    if xInd ==0 and yInd==0:
        return None

    imageScalerRes = prepareInputs(coaddTempExpDict_a.values(),
                                   coaddTempExpDict_a.keys(),
                                   assembleTask)
    if imageScalerRes is None:
        return None

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

    skyInfo = getSkyInfo(hits_skymap, xInd, yInd)

    coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
    coaddExposure.setCalib(assembleTask.scaleZeroPoint.getCalib())
    coaddExposure.getInfo().setCoaddInputs(assembleTask.inputRecorder.makeCoaddInputs())

    # remember to set metadata if you want any hope of running detection and measurement on this coadd:
    # self.assembleMetadata(coaddExposure, tempExpRefList, weightList)

    # most important thing is the psf
    coaddExposure.setFilter(coaddTempExpDict_a.values()[0].getFilter())
    coaddInputs = coaddExposure.getInfo().getCoaddInputs()

    for tempExp, weight in zip(coaddTempExpDict_a.values(), imageScalerRes.weightList):
        assembleTask.inputRecorder.addVisitToCoadd(coaddInputs, tempExp, weight)

    # takes numCcds as argument
    coaddInputs.ccds.reserve(ccdsinPatch)
    coaddInputs.visits.reserve(len(imageScalerRes.dataIdList))
    psf = measAlg.CoaddPsf(coaddInputs.ccds, coaddExposure.getWcs())
    coaddExposure.setPsf(psf)

    maskedImageList = afwImage.vectorMaskedImageF()
    coaddMaskedImage = coaddExposure.getMaskedImage()
    for dataId, imageScaler, exposure in zip(imageScalerRes.dataIdList,
                                             imageScalerRes.imageScalerList,
                                             coaddTempExpDict_a.values()):
        print dataId, imageScaler, exposure
        maskedImage = exposure.getMaskedImage()
        imageScaler.scaleMaskedImage(maskedImage)
        maskedImageList.append(maskedImage)

    maskedImage = afwMath.statisticsStack(maskedImageList,
                                          statsFlags, statsCtrl,
                                          imageScalerRes.weightList)

    coaddMaskedImage.assign(maskedImage, skyInfo.bbox)
    coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(),
                                coaddMaskedImage.getVariance())
    return coaddExposure


def makePatch(patchId, visit, dfgby_raveled, ccdsNeeded, hits_skymap):
    import lsst.afw.image as afwImg
    from lsst.pipe.tasks.makeCoaddTempExp import MakeCoaddTempExpTask, MakeCoaddTempExpConfig
    import lsst.coadd.utils as coaddUtils
    import numpy
    import lsst.skymap as skymap
    from lsst.meas.algorithms import CoaddPsf
    import lsst.skymap as skymap

    makeCTEConfig = MakeCoaddTempExpConfig()
    makeCTE = MakeCoaddTempExpTask(config=makeCTEConfig)

    xInd = patchId[0]
    yInd = patchId[1]

    skyInfo = getSkyInfo(hits_skymap, xInd, yInd)
    v = int(visit)
    coaddTempExp = afwImg.ExposureF(skyInfo.bbox, skyInfo.wcs)
    coaddTempExp.getMaskedImage().set(numpy.nan, afwImg.MaskU.getPlaneBitMask("NO_DATA"), numpy.inf)
    totGoodPix = 0
    didSetMetadata = False
    modelPsf = makeCTEConfig.modelPsf.apply() if makeCTEConfig.doPsfMatch else None
    setInputRecorder = False

    for b in dfgby_raveled:  # dfgby.get_group(a)[0].ravel():
        if not setInputRecorder:
            ccdsinPatch = len(dfgby_raveled)  # len(dfgby.get_group(a)[0].ravel())
            inputRecorder = makeCTE.inputRecorder.makeCoaddTempExpRecorder(v, ccdsinPatch)
            setInputRecorder = True
        numGoodPix = 0
        ccd = b
        if ccdsNeeded[(v, ccd)] is not None:
            calExp = ccdsNeeded[(v, ccd)].exposure
        else:
            continue

        ccdId = calExp.getId()
        warpedCcdExp = makeCTE.warpAndPsfMatch.run(calExp, modelPsf=modelPsf,
                                                   wcs=skyInfo.wcs, maxBBox=skyInfo.bbox).exposure
        if didSetMetadata:
            mimg = calExp.getMaskedImage()
            mimg *= (coaddTempExp.getCalib().getFluxMag0()[0] / calExp.getCalib().getFluxMag0()[0])
            del mimg

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
    inputRecorder.finish(coaddTempExp, totGoodPix)

    if totGoodPix > 0 and didSetMetadata:
        coaddTempExp.setPsf(modelPsf if makeCTEConfig.doPsfMatch else
                            CoaddPsf(inputRecorder.coaddInputs.ccds, skyInfo.wcs))

    return coaddTempExp


def matchCCDToPatch(calibRes, key):
    import lsst.skymap as skymap
    results = []
    ccd = key[1]
    visit = key[0]
    #print "ccd number: " + str(ccd)
    newSkyMapConfig = skymap.discreteSkyMap.DiscreteSkyMapConfig(projection='STG',
                                                                 decList=[-4.9325280994132905],
                                                                 patchInnerDimensions=[2000, 2000],
                                                                 radiusList=[4.488775723429071],
                                                                 pixelScale=0.333, rotation=0.0, patchBorder=100,
                                                                 raList=[154.10660740464786], tractOverlap=0.0)

    hits_skymap = skymap.discreteSkyMap.DiscreteSkyMap(config=newSkyMapConfig)
    tract = hits_skymap[0]

    if (calibRes is not None):
        exposure = calibRes.exposure
        bbox = exposure.getBBox()
        wcs = exposure.getWcs()
        corners = bbox.getCorners()
        xIndexMax, yIndexMax = tract.findPatch(wcs.pixelToSky(corners[0][0], corners[0][1])).getIndex()
        xIndexMin, yIndexMin = tract.findPatch(wcs.pixelToSky(corners[2][0], corners[2][1])).getIndex()
        yy = range(yIndexMin, yIndexMax + 1)
        xx = range(xIndexMin, xIndexMax + 1)

        for yIdx in yy:
            for xIdx in xx:
                results.append(( ccd, (xIdx, yIdx)))
                #print str(ccd) + ": " + str(xIdx) + ", " + str(yIdx)
    else:
        print "calibRes is none!"
        results.append((0,(0,0)))

    return results


def processCCDs(image):
    from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
    from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask

    calibRes = None
    # init tasks
    charImage = CharacterizeImageTask()
    calibrateConfig = CalibrateConfig(doPhotoCal=False, doAstrometry=False, doDeblend=False)
    calibrateTask = CalibrateTask(config=calibrateConfig)

    try:
        # characterize image
        charRes = charImage.characterize(image, exposureIdInfo=None, background=None)
        # calibrate image
        calibRes = calibrateTask.calibrate(charRes.exposure, exposureIdInfo=None, background=charRes.background,
                                           icSourceCat=None)
    except Exception as e:
        print "failed to calibrate the image"
        print str(e)

    return calibRes


def getListOfExposures(id, exposures, ccds):
    for ccd in range(1, ccds):
        key = tuple((id, str(ccd)))
        fn = "LSST/"+id + "/instcal" + id + "." + str(ccd) + ".fits"
        exposures[key] = fn


def download(s3fn, key):
    import lsst.afw.image as afwImg
    session = boto3.session.Session()
    s3 = session.resource('s3')
    fn = "instcal" + key[0] + "." + key[1] + ".fits"
    s3.meta.client.download_file('...', s3fn, fn)
    image = afwImg.ExposureF(fn)
    return image

def createCCDsInPatch(workers):
    ids = visits[0:1]
    ccds = 61
    exposuresLocation = {}
    for i in range(len(ids)):
        getListOfExposures(ids[i], exposuresLocation, ccds)
    i = 0
    expsDict = {}
    for key in exposuresLocation.keys():
        exposure = exposuresLocation[key]
        j = i % len(workers)
        r = e.submit(download, exposure, key, workers=[workers[j]])
        i += 1
        expsDict[key] = r
    calibCCDsDict = {}
    for key in expsDict.keys():
        calibCCDsDict[key] = e.submit(processCCDs, expsDict[key])
    visit = visits[0]
    ccdsPerPatch = []
    futureResults = []

    for ccd in range(1, ccds):
        key = (visit, str(ccd))
        r = e.submit(matchCCDToPatch, calibCCDsDict[key], key)
        futureResults.append(r)

        # Barrier 1: this is required to ensure all patches are ready and
        # mappped before create patch begins.
    for x in futureResults:
        ccdsPerPatch.extend(x.result())

    import cPickle
    f = open("ccdsInPatch2.p", 'wb')
    cPickle.dump(ccdsPerPatch,f)


def run(ids, workers, ccds):
    import lsst.afw.image as afwImg
    import lsst.skymap as skymap

    #create skymap
    newSkyMapConfig  = skymap.discreteSkyMap.DiscreteSkyMapConfig(projection='STG',
                                                                  decList=[-4.9325280994132905],
                                                                   patchInnerDimensions=[2000, 2000],
                                                                  radiusList=[4.488775723429071],
                                                                  pixelScale=0.333, rotation=0.0, patchBorder=100,
                                                                  raList=[154.10660740464786], tractOverlap=0.0)
    hits_skymap = skymap.discreteSkyMap.DiscreteSkyMap(config = newSkyMapConfig)
    tract = hits_skymap[0]

    # get list of exposures to download.
    exposuresLocation = {}
    for i in range(len(ids)):
        getListOfExposures(ids[i], exposuresLocation, ccds)
    i = 0
    expsDict = {}
    for key in exposuresLocation.keys():
        exposure = exposuresLocation[key]
        j = i % len(workers)
        r = e.submit(download, exposure, key, workers=[workers[j]])
        i += 1
        expsDict[key] = r

    #force download
    #expsDict = {k: v.result() for (k, v) in expsDict.iteritems()}

    # STEP 1: caliberate exposures via processCCDs
    calibCCDsDict = {}
    for key in expsDict.keys():
        calibCCDsDict[key] = e.submit(processCCDs, expsDict[key])


    """
    visit = visits[0]
    ccdsPerPatch = []
    futureResults = []

    # Step 2a: map each ccd to patches and groupby patchID.
    for ccd in range(1, ccds):
        key = (visit, str(ccd))
        r = e.submit(matchCCDToPatch, calibCCDsDict[key], key)
        futureResults.append(r)

    # Barrier 1: this is required to ensure all patches are ready and
    # mappped before create patch begins.
    for x in futureResults:
        ccdsPerPatch.extend(x.result())

    df = pd.DataFrame(ccdsPerPatch)
    dfgby = df.groupby(1)
    """
    import cPickle
    ccdsPerPatch = cPickle.load(open("ccdsInPatch2.p"))
    df = pd.DataFrame(ccdsPerPatch)
    dfgby = df.groupby(1)

    coaddTempExpDict = {}
    for patchid in dfgby.indices:
        coaddTempExpDict[patchid] = {}



    #Step 2b: Create patches from CCDs
    for v in ids:
        for patchId in dfgby.indices:
            ccdsNeeded = {(int(v), k): calibCCDsDict.get((v, str(k)), None)for k in dfgby.get_group(patchId)[0].ravel()}
            coaddTempExp = e.submit(makePatch, patchId, v, dfgby.get_group(patchId)[0].ravel(), ccdsNeeded, hits_skymap)
            coaddTempExpDict[patchId][int(v)] = coaddTempExp
            ccdsNeeded = None


    #Step 3:Stack patches, based on patch id
    stackedCoadds = {}
    for patchId in dfgby.indices:
        coaddExposure = e.submit(mergeCoadd, patchId, dfgby.get_group(patchId)[0].ravel(), coaddTempExpDict[patchId], hits_skymap)
        stackedCoadds[patchId] = coaddExposure

    #Step 4: Detect sources in each patch.
    detRes = []
    for patchId in dfgby.indices:
        detRes.append(e.submit(detect, stackedCoadds[patchId]))

    detRes = [x.result() for x in detRes]

if __name__ == '__main__':

    e = Executor("localhost:8786")
    start = datetime.datetime.now()
    workers = list(e.has_what())
    createCCDsInPatch(workers)


    ccds = 61
    for i in [2]:  # , 4, 8, 12, 24]:
        e.restart()
        workers = list(e.has_what())
        print("workers:", len(workers))
        run(visits[0:i], workers, ccds)
