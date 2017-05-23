#!/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/bin/python

# Desc:     Test setting environment variables in SciDB's stream()
# Author:   dzhao@cs.washington.edu
# Date:     3/14/2017

import sys
import os
import time

syspath_lsst = ['/home/ubuntu/obs_decam/python', '/home/ubuntu/lsst_stack/Linux64/mysqlpython/1.2.3.lsst2+3/lib/python', '/home/ubuntu/lsst_stack/Linux64/pipe_tasks/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/pykg_config/1.3.0/lib/python', '/home/ubuntu/lsst_stack/Linux64/healpy/1.8.1.lsst2+4/lib/python', '/home/ubuntu/lsst_stack/Linux64/skymap/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/coadd_chisquared/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/lmfit/0.9.3+3/lib/python', '/home/ubuntu/lsst_stack/Linux64/ip_diffim/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/ip_isr/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/meas_deblender/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/astrometry_net/0.67.123ff3e.lsst1/lib/python', '/home/ubuntu/lsst_stack/Linux64/meas_astrom/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/meas_algorithms/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/obs_test/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/pipe_base/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/pex_logging/12.1/python', '/home/ubuntu/lsst_stack/Linux64/coadd_utils/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/meas_base/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/esutil/0.6.0/lib/python', '/home/ubuntu/lsst_stack/Linux64/daf_butlerUtils/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/geom/12.1/python', '/home/ubuntu/lsst_stack/Linux64/skypix/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/stsci_distutils/0.3.7.lsst1+1/lib/python', '/home/ubuntu/lsst_stack/Linux64/python_d2to1/0.2.12.lsst2/lib/python', '/home/ubuntu/lsst_stack/Linux64/pyfits/3.4.0+6/lib/python', '/home/ubuntu/lsst_stack/Linux64/ndarray/1.2.0.lsst1/python', '/home/ubuntu/lsst_stack/Linux64/pex_config/12.1/python', '/home/ubuntu/lsst_stack/Linux64/afw/12.1+1/python', '/home/ubuntu/lsst_stack/Linux64/daf_persistence/12.1/python', '/home/ubuntu/lsst_stack/Linux64/pyyaml/3.11.lsst1+2/lib/python', '/home/ubuntu/lsst_stack/Linux64/daf_base/12.1/python', '/home/ubuntu/lsst_stack/Linux64/pex_policy/12.1/python', '/home/ubuntu/lsst_stack/Linux64/utils/12.1/python', '/home/ubuntu/lsst_stack/Linux64/python_psutil/4.1.0+2/lib/python', '/home/ubuntu/lsst_stack/Linux64/log/12.1/python', '/home/ubuntu/lsst_stack/Linux64/pex_exceptions/12.1/python', '/home/ubuntu/lsst_stack/Linux64/python_future/0.15.2+1/lib/python', '/home/ubuntu/lsst_stack/Linux64/sconsUtils/12.1/python', '/home/ubuntu/lsst_stack/Linux64/base/12.1/python', '/home/ubuntu/lsst_stack/eups/python', '/home/ubuntu/lsst_stack/Linux64/mysqlpython/1.2.3.lsst2+3/lib/python/MySQL_python-1.2.3-py2.7-linux-x86_64.egg', '/home/ubuntu/lsst_stack/Linux64/healpy/1.8.1.lsst2+4/lib/python/healpy-1.8.1-py2.7-linux-x86_64.egg', '/home/ubuntu/lsst_stack/Linux64/lmfit/0.9.3+3/lib/python/lmfit-0.9.3-py2.7.egg', '/home/ubuntu/lsst_stack/Linux64/stsci_distutils/0.3.7.lsst1+1/lib/python/stsci.distutils-0.3.7-py2.7.egg', '/home/ubuntu/lsst_stack/Linux64/python_d2to1/0.2.12.lsst2/lib/python/d2to1-0.2.12.post1-py2.7.egg', '/home/ubuntu/lsst_stack/Linux64/pyfits/3.4.0+6/lib/python/pyfits-3.4-py2.7-linux-x86_64.egg', '/home/ubuntu/lsst_stack/Linux64/python_psutil/4.1.0+2/lib/python/psutil-4.1.0-py2.7-linux-x86_64.egg', '/home/ubuntu/lsst_stack/Linux64/python_future/0.15.2+1/lib/python/future-0.15.2-py2.7.egg', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python27.zip', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python2.7', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python2.7/plat-linux2', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python2.7/lib-tk', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python2.7/lib-old', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python2.7/lib-dynload', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python2.7/site-packages', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg', '/home/ubuntu/lsst_stack/Linux64/miniconda2/3.19.0.lsst4/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg', '/home/ubuntu/lsst_stack/Linux64/ndarray/1.2.0.lsst1/python']
sys.path = sys.path + syspath_lsst 

######################## Begin
#Parmita's change:
#import cPickle
#path = cPickle.load(open("/home/ubuntu/scidb_lsst/path.p"))
#rest = sys.path
#sys.path = path+rest

#import getpass
#u = str(getpass.getuser())
#p=sys.path
#q = '\n'.join(map(str, p))
#sys.stderr.write("\n\n=====> DFZ is back! "+time.ctime()+": Let's start. \n")
#sys.stderr.write("\n\n**** path: "+str(q)+"\n")
#sys.stderr.write("\n\n*** user : " + u+ "\n")
######################### End

#os.system('source /home/ubuntu/lsst_stack/loadLSST.bash')
#os.system('setup lsst_distrib')


#import lsst.afw.image as afwImage
#from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
#from lsst.pipe.tasks.calibrate import CalibrateTask, CalibrateConfig
#import lsst.afw.image.basicUtils as bu
#import lsst.skymap as skymap
#import lsst.daf.persistence as dafPersist

sys.stderr.write("\n\n=====> DFZ " + time.ctime() + ": started...\n")

pid = str(os.getpid())
envs = str(os.environ)

end_of_interaction = 0
while (end_of_interaction != 1):
    header = sys.stdin.readline().rstrip()
    if(header != "0"):
        num_lines = int(header)  #how many lines did we get?
        for i in range(0, num_lines):
            _ = sys.stdin.readline().rstrip()

        print(0)
        sys.stdout.flush()

    else:
        end_of_interaction = 1

        print(1)
        print("pid = " + pid + " done, envs = " + envs)
        sys.stdout.flush()

sys.stderr.write("\n\n=====> DFZ " + time.ctime() + ": Done. \n")
