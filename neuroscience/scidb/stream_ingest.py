from scidbpy import connect
import numpy as np
import nibabel as nib

sdb = connect('http://localhost:8080')


data_path = '/home/ubuntu/mri_data/100307/'
img = nib.load(data_path + '/data.nii.gz')
data = img.get_data()


data_sdb_p0 = sdb.from_array(data[:,:,:,0:144], chunk_size = (145,174,145,9), persistent=True)
data_sdb_p1 = sdb.from_array(data[:,:,:,144:288], chunk_size = (145,174,145,9), persistent=True)

print "Done"
