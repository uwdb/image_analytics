
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import nibabel as nib
import subprocess
import os
from os import system
#import dipy.core.gradients as dpg
#from dipy.segment.mask import median_otsu
#from dipy.reconst import dti
from scidbpy import connect
#from scidbpy import robust
#from scidbpy import save

sdb = connect('http://localhost:8080')

#example: create a new array
#sdb.query("create_array(denoised_array, <f0:float NOT NULL> [i0=0:144,145,0,i1=0:173,174,0,i2=0:144,145,0,i3=0:143,18,0], false)")
print str(time.ctime()) + ": started"

data_sdb_p0 = sdb.wrap_array('original_half_1st')
print str(time.ctime()) + ": chunk 0 loaded as " + data_sdb_p0.name
#data_np_p0 = data_sdb_p0.toarray()
#print str(time.ctime()) + ": chunk 0 converted to numpy array"

data_sdb_p1 = sdb.wrap_array('original_half_2nd')
print str(time.ctime()) + ": chunk 1 loaded"
#data_np_p1 = data_sdb_p1.toarray()
#print str(time.ctime()) + ": chunk 1 converted to numpy array"

fitdata_p0 = sdb.concatenate((data_sdb_p0, data_sdb_p1), axis=3)
print str(time.ctime()) + ": fit chunk 0 concated"

sdb.query("store("+fitdata_p0.name +", fit_subject)")
exit(0)

fitdata_sdb_p0 = sdb.from_array(fitdata_p0, persistent=True)
print str(time.ctime()) + ": fit chunk 0 persisted"

fitdata_p1 = sdb.concatenate((data_isdb_p0[:,87:174,:,:], data_sdb_p1[:,87:174,:,:]), axis=3)
print str(time.ctime()) + ": fit chunk 1 concated"
fitdata_sdb_p1 = sdb.from_array(fitdata_p1, persistent=True)
print str(time.ctime()) + ": fit chunk 1 persisted"

exit(0)

data_sdb = sdb.concatenate((data_sdb_p0, data_sdb_p1), axis=3)
print str(time.ctime()) + ": two chunks concated\n "

print "data_sdb.name =" + data_sdb.name
print "data_sdb.shape = " + str(data_sdb.shape)

sdb.query("store("+ data_sdb.name +", data_to_fit)")
print str(time.ctime()) + ": array persisted\n "

#sdb.query("store("+ data_fit_original_1st.name +", data_to_fit_1st)")
#data_fit_original_2nd = sdb.from_array(data_np[:,:,:,144:288])

exit(0)

sdb.query("store("+ data_sdb.name +", original_subject)")


array_persist = sdb.wrap_array(data_sdb.name, persistent=True)
print str(time.ctime()) + ": array persisted\n "

exit(0)

array_b = sdb.wrap_array("B") 
array_c = sdb.wrap_array("Cplus") 

print "array_b = \n"
print array_b.toarray()

print "array_c = \n"
print array_c.toarray()


#array_d = 
sdb.concatenate((array_b, array_c), axis=1)
#print "array_d = "
#print array_d.toarray()

instances = sdb.afl.list('instances')

print array_b.shape

X = np.random.random((5, 4))
Y = np.random.random((3, 4))
Y = X
print Y
new_array = sdb.from_array(X, persistent=True)
print "new_array.name = ", new_array.name

x = 0
if x > 1:
    y = 1
else:
    x = 0



