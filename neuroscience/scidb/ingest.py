#Ingest 25 subjects from disk to scidb
import warnings
warnings.filterwarnings("ignore")
import time
#import numpy as np
import nibabel as nib
import subprocess
import os
from os import system
#import dipy.core.gradients as dpg
#from dipy.segment.mask import median_otsu
#from dipy.reconst import dti
from scidbpy import connect

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
        system('mkdir -p ' + local_path + subject_id)
        system(cmd_aws + subject_id + '  ' + local_path + subject_id)
        img = nib.load(local_path + subject_id + '/data.nii.gz')
        data = img.get_data()

###########################
#DFZ: ingestion for denoise
###########################
        data_sdb_p0 = sdb.from_array(data[:,:,:,0:144], chunk_size=(145,174,145,18), persistent=True)
        sdb.query("rename(" + data_sdb_p0.name + ", original_half_1st)")
        print str(time.ctime()) + ": chunk 0 loaded"
        data_sdb_p1 = sdb.from_array(data[:,:,:,144:288], chunk_size=(145,174,145,18), persistent=True)
        sdb.query("rename(" + data_sdb_p1.name + ", original_half_2nd)")
        print str(time.ctime()) + ": chunk 1 loaded"
        #data_sdb = sdb.concatenate((data_sdb_p0, data_sdb_p1), axis=3)
        #print str(time.ctime()) + ": two chunks concated\n "
        #sdb.query("store("+ data_sdb.name +", original_subject)")
        #array_persist = sdb.wrap_array(data_sdb.name, persistent=True)
        #print str(time.ctime()) + ": array persisted\n "
#DFZ: don't bother to merge them before denoising because it's hard to merge+persistent
        

##############################
#DFZ: ingestion for model-fit:
##############################
#        data_sdb_p0 = sdb.from_array(data[:,0:87,:,:], chunk_size=(29,29,29,288), persistent=True)
#        data_sdb_p1 = sdb.from_array(data[:,87:174,:,:], chunk_size=(29,29,29,288), persistent=True)

#        print "start to concatenate..."
#        data_sdb = sdb.concatenate((data_sdb_p0, data_sdb_p1), axis=3)
#        print "start to persistent..." #this is very slow
#        data_sdb_disk = data_sdb.rename("sub_full", persistent=True)
        #data_sdb_persist = sdb.wrap_array(data_sdb)#this is buggy
        ingest_o.write('%.3f\n' % (time.time() - tm_start))
        ingest_o.close()

        #subjects_sdb[cnt] = data_sdb #Keep all original subjects in SciDB
#        system('rm -rf ' + local_path + subject_id) #Remove original files
        cnt += 1
        print '\n=====> Completed %d / %s: %s\n' % (cnt, n_subjects, time.ctime())
print "Ingestion done:", str(time.ctime())
