# Desc:     ingest LSST data into SciDB for stream() processing
# Author:   dzhao@cs.washington.edu
# Date:     3/5/2017

import lsst.afw.image as afwImage
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
from astropy.io import fits
import lsst.afw.image.basicUtils as bu
import lsst.skymap as skymap
import numpy as np
import os
import lsst.daf.persistence as dafPersist

path_rawdata = os.path.expanduser("~") + "/lsst_data/raw"
butler = dafPersist.Butler(path_rawdata)

#visits = ["0289697", "0289783"]
visits = ["0289697"]
ccd_ids = [1, 2]
#for each visit
#   for each ccd
#       write the array into csv<visit_id, ccd_id, data>
csv_file = open(path_rawdata + '/lsst.csv', 'w')
for visit in visits:
    for ccd_id in ccd_ids:
        exposure = butler.get("instcal", visit=int(visit), ccdnum=ccd_id, filter='g', immediate=True)
        image_array = exposure.getMaskedImage().getImage().getArray()
        print(str(image_array.shape)+"\n")
        
        for i in range(0,len(image_array)):
            for j in range(0,len(image_array[0])):
                csv_file.write(visit+","+str(ccd_id)+","+str(i)+","+str(j)+","+str(image_array[i][j])+"\n")
#        fname = path_rawdata + "crblasted" + visit + "/instcal" + "visit." + str(ccd_id) + ".fits"
#        fits_header = afwImage.readMetadata(fname)
#        wcs = afwImage.makeWcs(fits_header)
csv_file.close()

print "all done\n"
