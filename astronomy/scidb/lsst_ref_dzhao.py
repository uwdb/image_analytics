import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.processCcd import ProcessCcdTask
from lsst.obs.decam.decamNullIsr import DecamNullIsrTask
from lsst.ip.isr import IsrTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
from lsst.pipe.tasks.makeCoaddTempExp import MakeCoaddTempExpTask, MakeCoaddTempExpConfig
import lsst.afw.image as afwImage
import lsst.coadd.utils as coaddUtils
from lsst.meas.algorithms import CoaddPsf
import lsst.afw.image       as afwImg
import pandas as pd
import lsst.meas.algorithms as measAlg
import lsst.skymap as skymap
import numpy
import datetime
from lsst.pipe.tasks.assembleCoadd import AssembleCoaddTask, AssembleCoaddConfig
import lsst.afw.math as afwMath
from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask, DetectCoaddSourcesConfig

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
def getTempExpDatasetName():
    return 'Deep'


def prepareInputs(cteList, dataIdList, assembleTask):
    """!
    \brief Prepare the input warps for coaddition by measuring the weight for each warp and the scaling
    for the photometric zero point.

    Each coaddTempExp has its own photometric zeropoint and background variance. Before coadding these
    coaddTempExps together, compute a scale factor to normalize the photometric zeropoint and compute the
    weight for each coaddTempExp.

    \param[in] refList: List of data references to tempExp
    \return Struct:
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
    newDataIdList = [] #make clean list incase scaling failed. output lists should all be same length
    weightList = []
    imageScalerList = []
    tempExpName = getTempExpDatasetName()
    for dataId, tempExp in zip(dataIdList, cteList):
        maskedImage = tempExp.getMaskedImage()
        imageScaler = assembleTask.scaleZeroPoint.computeImageScaler(
            exposure = tempExp,
            dataRef = None,
        )
        try:
            pass
            # You can comment this out
            #imageScaler.scaleMaskedImage(maskedImage) #warning. This changes the images!
        except Exception as e:
            print("Scaling failed for %s (skipping it): %s" % (tempExpRef.dataId, e))
            continue
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

def main():
    # try out one exposure
    #visits = ["0288935","0288976"] #,"0289893","0289913","0289931","0289614","0289818","0289820", "0289850","0289851","0289871","0289892", "0288935","0288976","0289016","0289056","0289161","0289202","0289243","0289284","0289368","0289409","0289450","0289493","0289573","0289656"]
    visits = ["0288976","0288935"] 
    ccds = []
    exit(0)
    for i in range(1,61):
           ccds.append(i)

    filterName='g'

    DATA_PATH = "/root/extra_home/lsst_data/"
    #spathprefix = "/home/dongfang/download/lsst_data/"
    spathprefix = DATA_PATH + "raw/"
    #calexpsloc = "/home/dongfang/download/lsst_data/calexps/"
    calexpsloc = DATA_PATH + "calexps/"
    #coaddloc = "/home/dongfang/download/lsst_data/coadds/"
    coaddloc = DATA_PATH + "coadds/"
    #mergecoaddloc = "/home/dongfang/download/lsst_data/merge/"
    mergecoaddloc = DATA_PATH + "merge/"


    # Characterize Image
    charImageConfig = CharacterizeImageConfig()
    charImage = CharacterizeImageTask()

    calibrateConfig = CalibrateConfig(doPhotoCal=False, doAstrometry=False)
    calibrateTask = CalibrateTask(config=calibrateConfig)

    makeCTEConfig = MakeCoaddTempExpConfig()
    makeCTE = MakeCoaddTempExpTask(config=makeCTEConfig)


    newSkyMapConfig  = skymap.discreteSkyMap.DiscreteSkyMapConfig(projection='STG',
                                                                  decList=[-4.9325280994132905],
                                                                   patchInnerDimensions=[2000, 2000],
                                                                  radiusList=[4.488775723429071],
                                                                  pixelScale=0.333, rotation=0.0, patchBorder=100,
                                                                  raList=[154.10660740464786], tractOverlap=0.0)

    hits_skymap = skymap.discreteSkyMap.DiscreteSkyMap(config = newSkyMapConfig)
    tract = hits_skymap[0]
    coaddTempDict = {}
    calibResDict={}
    f = open("log.txt", 'wb')
    start = datetime.datetime.now()
    #process CCDs to create calexps.
    for v in visits:
        for ccd in ccds:
            visit = int(v)
            filename =   "instcal"+v+"."+str(ccd)+".fits"
            calexpfn = calexpsloc+v+"/"+filename
            source = spathprefix+ v+"/"+filename
            exposure = afwImg.ExposureF(source)

            try:
                # Characterize Image
                charRes = charImage.characterize(exposure, exposureIdInfo=None,  background=None)
            except:
                f.write("DFZ DEBUG at charRes: errors in visit " + v + ", ccd " + str(ccd) + "\n")

            try:
                # Caliberate Image
                calibRes = calibrateTask.calibrate(charRes.exposure,
                                    exposureIdInfo=None, background=charRes.background, icSourceCat=None)
            except:
                f.write("DFZ DEBUG at calibRes: errors in visit " + v + ", ccd " + str(ccd) + "\n")

            try:
                #write out calexps
                calibRes.exposure.writeFits(calexpfn)
                #calbresDict.append((v,ccd),calibRes)
            except:
                f.write("DFZ DEBUG at calibRes.exposure: errors in visit " + v + ", ccd " + str(ccd) + "\n")

    end = datetime.datetime.now()
    d = end-start

    f.write("time for creating calexps: " )
    f.write (str(d.total_seconds()))
    f.write("\n")


    #time for creating co-add tempexps.
    start = datetime.datetime.now()

    # map calexps to patch-ids
    visit = visits[0]
    ccdsPerPatch =[]

    for ccd in ccds:
        filename  = "instcal"+visit+"."+str(ccd)+".fits"
        source   = calexpsloc + visit+ "/"+filename
        exposure = afwImg.ExposureF(source)
        bbox = exposure.getBBox()
        wcs = exposure.getWcs()
        corners = bbox.getCorners()
        xIndexMax, yIndexMax = tract.findPatch(wcs.pixelToSky(corners[0][0], corners[0][1])).getIndex()
        xIndexMin, yIndexMin = tract.findPatch(wcs.pixelToSky(corners[2][0], corners[2][1])).getIndex()
        yy = range(yIndexMin, yIndexMax+1)
        xx = range(xIndexMin, xIndexMax+1)

        for yIdx in yy:
            for xIdx in xx:
                ccdsPerPatch.append((ccd,(xIdx,yIdx)))
        print len(ccdsPerPatch)
    #import cPickle
    #cPickle.dump(open("ccdsinpatch.p",'wb'),ccdsPerPatch)

    # import cPickle
    # f = open("ccdsInPatch.p",'wb')
    # cPickle.dump(ccdsInPatch,f)
    #import cPickle

    #ccdsInPatch = cPickle.load(open("ccdsInPatch.p",'rb'))
    df = pd.DataFrame(ccdsPerPatch)

    dfgby  = df.groupby(1)
    makeCTEConfig = MakeCoaddTempExpConfig()
    makeCTE = MakeCoaddTempExpTask(config=makeCTEConfig)
    coaddTempExpDict = {}
    for visit in visits:
        for a in dfgby.indices:
            coaddTempExpDict[a] = {}
            xInd = a[0]
            yInd = a[1]
            skyInfo = getSkyInfo(hits_skymap,xInd,yInd)
            v = int(visit)

            coaddTempExp = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
            coaddTempExp.getMaskedImage().set(numpy.nan, afwImage.MaskU.getPlaneBitMask("NO_DATA"), numpy.inf)
            totGoodPix = 0
            didSetMetadata = False
            modelPsf = makeCTEConfig.modelPsf.apply() if makeCTEConfig.doPsfMatch else None
            setInputRecorder=False

            for b in dfgby.get_group(a)[0].ravel():
                print a
                print b
                if not setInputRecorder :
                    ccdsinPatch = len(dfgby.get_group(a)[0].ravel())
                    try:
                        inputRecorder =  makeCTE.inputRecorder.makeCoaddTempExpRecorder(v, ccdsinPatch)
                    except:
                        f.write("DFZ DEBUG at inputRecorder\n")
                    setInputRecorder=True
                numGoodPix = 0
                ccd = b
                filename  = "instcal"+visit+"."+str(ccd)+".fits"
                source   = calexpsloc + visit+ "/"+filename
                calExp = afwImg.ExposureF(source)
                ccdId = calExp.getId()
                warpedCcdExp = makeCTE.warpAndPsfMatch.run(calExp, modelPsf=modelPsf,
                                                           wcs=skyInfo.wcs,maxBBox=skyInfo.bbox).exposure
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

            coaddTempExpDict[a][v] = coaddTempExp
            coaddfilename = coaddloc+visit+"/"+"instcal"+visit+"."+str(xInd)+"_"+str(yInd)+".fits"
            coaddTempExp.writeFits(coaddfilename)

    end = datetime.datetime.now()
    d = end-start
    f.write("time for creating co-add tempexps:\n " )
    f.write (str(d.total_seconds()))
    f.write("\n")

    #DFZ: stop here
    exit(0)

    start = datetime.datetime.now()

    config = AssembleCoaddConfig()
    assembleTask = AssembleCoaddTask(config=config)
    mergcoadds = {}
    for a in dfgby.indices:
        ccdsinPatch = len(dfgby.get_group(a)[0].ravel())
        xInd = a[0]
        yInd = a[1]

        imageScalerRes = prepareInputs(coaddTempExpDict[a].values(),
                                   coaddTempExpDict[a].keys(),
                                   assembleTask)
        mask = None
        doClip=False
        if mask is None:
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

        if doClip:
            statsFlags = afwMath.MEANCLIP
        else:
            statsFlags = afwMath.MEAN

        coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddExposure.setCalib(assembleTask.scaleZeroPoint.getCalib())
        coaddExposure.getInfo().setCoaddInputs(assembleTask.inputRecorder.makeCoaddInputs())

        #remember to set metadata if you want any hope of running detection and measurement on this coadd:
        #self.assembleMetadata(coaddExposure, tempExpRefList, weightList)

        #most important thing is the psf
        coaddExposure.setFilter(coaddTempExpDict[a].values()[0].getFilter())
        coaddInputs = coaddExposure.getInfo().getCoaddInputs()

        for tempExp, weight in zip(coaddTempExpDict[a].values(), imageScalerRes.weightList):
            assembleTask.inputRecorder.addVisitToCoadd(coaddInputs, tempExp, weight)

        #takes numCcds as argument


        coaddInputs.ccds.reserve(ccdsinPatch)
        coaddInputs.visits.reserve(len(imageScalerRes.dataIdList))
        psf = measAlg.CoaddPsf(coaddInputs.ccds, coaddExposure.getWcs())
        coaddExposure.setPsf(psf)


        maskedImageList = afwImage.vectorMaskedImageF()
        coaddMaskedImage = coaddExposure.getMaskedImage()
        for dataId, imageScaler, exposure in zip(imageScalerRes.dataIdList,
                                                 imageScalerRes.imageScalerList,
                                                 coaddTempExpDict[a].values()):
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

        # write out Coadd!
        mergefilename  = mergecoaddloc+ str(xInd)+"_"+str(yInd)+".fits"
        mergcoadds[a] = coaddExposure
        coaddExposure.writeFits(mergefilename)

    end = datetime.datetime.now()
    d = end-start
    f.write("time for creating merged co-adds:\n " )
    f.write (str(d.total_seconds()))
    f.write("\n")

    start = datetime.datetime.now()
    config = DetectCoaddSourcesConfig()
    detectCoaddSources = DetectCoaddSourcesTask(config=config)
    for a in dfgby.indices:

        # Detect on Coadd
        exp = mergcoadds[a]
        detRes = detectCoaddSources.runDetection(exp, idFactory=None)

    end = datetime.datetime.now()
    d = end-start
    f.write("time for detecting sources:\n ")
    f.write (str(d.total_seconds()))
    f.close()

if __name__ =='__main__':
        main()
