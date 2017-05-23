#Desc:     Wrapper call for streaming calibration of CCDs
#Author:   dzhao@cs.washington.edu
#Date:     3/5/2017
#which python
#source /home/ubuntu/lsst_stack/loadLSST.bash
iquery -anq "load_library('stream')"
iquery -aq "stream(IMAGE_ARRAY, '
    source /home/ubuntu/lsst_stack/loadLSST.bash; 
    export TAG=v12_1_2; 
    setup -r /home/ubuntu/obs_decam -t $TAG; 
    export EUPS_DIR=/home/ubuntu/lsst_stack/eups;
    $EUPS_DIR/bin/eups_setup DYLD_LIBRARY_PATH=/home/ubuntu/lsst_stack/Linux64/healpy/1.8.1.lsst2+4/lib:/home/ubuntu/lsst_stack/Linux64/skymap/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/coadd_chisquared/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/ip_diffim/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/ip_isr/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/meas_deblender/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/astrometry_net/0.67.123ff3e.lsst1/lib:/home/ubuntu/lsst_stack/Linux64/meas_astrom/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/meas_algorithms/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/pex_logging/12.1/lib:/home/ubuntu/lsst_stack/Linux64/coadd_utils/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/meas_base/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/gsl/1.16.lsst3/lib:/home/ubuntu/lsst_stack/Linux64/minuit2/5.34.14/lib:/home/ubuntu/lsst_stack/Linux64/wcslib/5.13.lsst1/lib:/home/ubuntu/lsst_stack/Linux64/cfitsio/3360.lsst4/lib:/home/ubuntu/lsst_stack/Linux64/fftw/3.3.4.lsst2/lib:/home/ubuntu/lsst_stack/Linux64/pex_config/12.1/lib:/home/ubuntu/lsst_stack/Linux64/afw/12.1+1/lib:/home/ubuntu/lsst_stack/Linux64/daf_persistence/12.1/lib:/home/ubuntu/lsst_stack/Linux64/daf_base/12.1/lib:/home/ubuntu/lsst_stack/Linux64/pex_policy/12.1/lib:/home/ubuntu/lsst_stack/Linux64/mariadbclient/10.1.11.lsst3/lib:/home/ubuntu/lsst_stack/Linux64/utils/12.1/lib:/home/ubuntu/lsst_stack/Linux64/apr_util/1.5.4/lib:/home/ubuntu/lsst_stack/Linux64/apr/1.5.2/lib:/home/ubuntu/lsst_stack/Linux64/log4cxx/0.10.0.lsst6+1/lib:/home/ubuntu/lsst_stack/Linux64/log/12.1/lib:/home/ubuntu/lsst_stack/Linux64/boost/1.60.lsst1/lib:/home/ubuntu/lsst_stack/Linux64/pex_exceptions/12.1/lib:/home/ubuntu/lsst_stack/Linux64/base/12.1/lib eups -r $EUPS_DIR;
    setup miniconda2; 
    setup lsst;
    /home/ubuntu/scidb_lsst/stream_env_set.py')"
