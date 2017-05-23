#Ingest 25 subjects from disk to scidb
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import nibabel as nib
import subprocess
import os
from os import system
import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu
from dipy.reconst import dti
from scidbpy import connect
from scidbpy import robust
from scidbpy import save

home_path = os.path.expanduser("~")
cmd_aws = 'aws s3 sync s3://imagedb-data/'
local_path = home_path + '/download/subjects/'
n_subjects = subprocess.check_output('cat subjects.file | wc -l', shell=True).strip()
n_nodes = subprocess.check_output('cat hosts.txt | wc -l', shell=True).strip()
system('mkdir -p ' + local_path)
system('rm -rf ' + local_path + '*')
system('rm -f *.output_' + n_subjects+'_'+n_nodes)

print "Start loading data:", str(time.ctime())

sdb = connect('http://localhost:8080')
subjects_sdb = {} 
cnt = 0
with open('subjects.file') as f:
    for line in f:
        subject_id = line.strip()

        ingest_o = open('ingest.output_'+n_subjects+'_'+n_nodes, 'a')
        tm_start = time.time()
        system(cmd_aws + subject_id + '  ' + local_path + subject_id)
        img = nib.load(local_path + subject_id + '/data.nii.gz')
        data = img.get_data()
        data_sdb_p0 = sdb.from_array(data[0:72,:,:,:])
        data_sdb_p1 = sdb.from_array(data[72:145,:,:,:])
        data_sdb = sdb.concatenate((data_sdb_p0, data_sdb_p1), axis=0)
        ingest_o.write('%.3f\n' % (time.time() - tm_start))
        ingest_o.close()

        filter_o = open('filter.output_'+n_subjects+'_'+n_nodes, 'a')
        tm_start = time.time()
        gtab = dpg.gradient_table(local_path+subject_id+'/bvals', local_path+subject_id+'/bvecs', b0_threshold=10)
        data_filtered = data_sdb.compress(sdb.from_array(gtab.b0s_mask), axis=3)
        filter_o.write('%.3f\n' % (time.time() - tm_start))
        filter_o.close()

        mean_o = open('mean.output_'+n_subjects+'_'+n_nodes, 'a')
        tm_start = time.time()
        mean_b0_sdb = data_filtered.mean(index=3)
        mean_o.write('%.3f\n' % (time.time() - tm_start))
        mean_o.close()

        #subjects_sdb[cnt] = data_sdb #Keep all original subjects in SciDB
#        system('rm -rf ' + local_path + subject_id) #Remove original files
        cnt += 1
        print '\n=====> Completed %d / %s: %s\n' % (cnt, n_subjects, time.ctime())
print "Ingestion done:", str(time.ctime())
