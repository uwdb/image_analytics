#Ingest 25 subjects from disk to scidb
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import nibabel as nib
import os
from os import system

cmd_aws = 'aws s3 sync s3://imagedb-data/'
home_path = os.path.expanduser("~")
res_fname = './ingest_aio.result'
system('rm -f ' + res_fname)

with open('subjects.file') as f:
    for line in f:
        subject_id = line.strip()
        local_path = home_path + '/mri_data/' + subject_id + '/' 
        system('mkdir -p ' + local_path)
        outfile = open(res_fname, 'a')

        tm_start = time.time()

        #download and reformat data
        system(cmd_aws + subject_id + '  ' + local_path)
        img = nib.load(local_path + '/data.nii.gz')
        raw_data = img.get_data()
        np_data = np.frombuffer(raw_data)
        data = np_data.reshape(-1,288)
        np.savetxt('/run/shm/mri.csv', data, delimiter=',', fmt='%1.4e')

        tm_mid = time.time()

        #aio_input
        system('iquery -f ingest_aio.aql')

        outfile.write('%s, %.3f, %.3f\n' % (subject_id, tm_mid-tm_start, time.time()-tm_mid))
        outfile.close()

        system('rm -rf ' + local_path)
