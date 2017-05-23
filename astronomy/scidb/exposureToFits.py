from __future__ import print_function

import lsst.afw.image as afwImage
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
from astropy.io import fits
import lsst.afw.image.basicUtils as bu
import lsst.skymap as skymap
import lsst.daf.persistence as dafPersist
import numpy as np
import os
import sys

import lsst.log
logger = lsst.log.Log.getDefaultLogger()
logger.setLevel(lsst.log.FATAL)

HOME_PATH = os.path.expanduser("~")

fn = HOME_PATH + "/lsst_data/raw/crblasted0289697/instcal0289697.12.fits"
image = afwImage.ExposureF(fn)
charImage = CharacterizeImageTask()
charRes = charImage.characterize(image, exposureIdInfo=None, background=None)

# try out one exposure
visit=289697
ccdnum = 1
filterName='g'

"""
folder data should have the following structure - for every visit there should be folder <visitnumber> and <crblastedvisitnumber>
<visitnumber> should have three files: dqmask0288935.fits.fz    instcal0288935.fits.fz  wtmap0288935.fits.fz, where visit number is 0288935
<crblasted> should have one file per ccd called instcal0288935.1.fits where visit number is 0288935 and ccd number is 1

All fo this data is available at:https://lsst-web.ncsa.illinois.edu/~yusra/hits_calexp_dir/
root folder in this case data, should also have a file _mapper with the following line.

  lsst.obs.decam.DecamMapper
"""
##############################
#Initial conversion to arrays#
##############################

#DFZ NOT WORKING!: try to suppress the stdout from LSST, because stdout is used in SciDB for data streaming
sys.stdout = open('lsst_out.log', 'w+')
sys.stderr = open('lsst_err.log', 'w+')

#DFZ NOT WORKING!: try, again..., to suppress debug messages from LSST to stdout 
#from contextlib import redirect_stdout, merged_stderr_stdout
#stdout_redirected(to=os.devnull)
#merged_stderr_stdout()

butler = dafPersist.Butler(HOME_PATH + '/lsst_data/raw')
exposure = butler.get("instcal", visit=visit, ccdnum=ccdnum, filter='g', immediate=True)
exposureIdInfo = butler.get("expIdInfo", visit=visit, ccdnum=ccdnum, filter='g', immediate=True) #DFZ: probably useless

image = exposure.getMaskedImage().getImage().getArray()
mask  = exposure.getMaskedImage().getMask().getArray()
variance = exposure.getMaskedImage().getVariance().getArray()
exposure.getMetadata() #DFZ: what is this?


### you will need the Wcs object in addition to the
filename = HOME_PATH + "/lsst_data/raw/crblasted0289697/instcal0289697.1.fits"
fitsHeader = afwImage.readMetadata(filename)
wcs = afwImage.makeWcs(fitsHeader)

maskedImage =  bu.makeMaskedImageFromArrays(image, mask, variance)
image2 = afwImage.ExposureF(maskedImage)
image2.setWcs(wcs)

###run charecterize and caliberate

charRes = charImage.characterize(image2, exposureIdInfo=None, background=None)
calibrateConfig = CalibrateConfig(doPhotoCal=False, doAstrometry=False)
calibrateTask = CalibrateTask(config=calibrateConfig)
calibRes = calibrateTask.calibrate(charRes.exposure, exposureIdInfo=None, background=charRes.background, icSourceCat=None)

### ### extract arrays from processed exposure###


image = calibRes.exposure.getMaskedImage().getImage().getArray()
mask = calibRes.exposure.getMaskedImage().getMask().getArray()
variance = calibRes.exposure.getMaskedImage().getVariance().getArray()
maskedImage =  bu.makeMaskedImageFromArrays(image, mask, variance)


##recreate exposure
image3 = afwImage.ExposureF(maskedImage)
image3.setWcs(wcs)


# run next step: create patch
newSkyMapConfig = skymap.discreteSkyMap.DiscreteSkyMapConfig(projection='STG',
                                                             decList=[-4.9325280994132905],
                                                             patchInnerDimensions=[2000, 2000],
                                                             radiusList=[4.488775723429071],
                                                             pixelScale=0.333, rotation=0.0, patchBorder=100,
                                                             raList=[154.10660740464786], tractOverlap=0.0)

hits_skymap = skymap.discreteSkyMap.DiscreteSkyMap(config=newSkyMapConfig)
tract = hits_skymap[0]

bbox = image3.getBBox()
wcs = image3.getWcs()
corners = bbox.getCorners()
xIndexMax, yIndexMax = tract.findPatch(wcs.pixelToSky(corners[0][0], corners[0][1])).getIndex()
xIndexMin, yIndexMin = tract.findPatch(wcs.pixelToSky(corners[2][0], corners[2][1])).getIndex()


#DFZ: try to reset stdout
print("You should NOT see me")
sys.stdout = sys.__stdout__
#stdout_redirected()
#merged_stderr_stdout()
print("\n\n===============================")
print("image.shape = ", image.shape)
print("mask.shape = ", mask.shape)
print("variance.shape = ", variance.shape)
print("===============================\n\n")
