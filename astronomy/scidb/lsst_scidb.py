data_path = '/home/dongfang/download/lsst_data/'
visits = ["0288935", "0288976"]

visit = visits[0]
ccd_id = '1' #integer between 1 and 60

from astropy.io import fits
hdulist = fits.open(data_path + visit + '/instcal' + visit + '.' + ccd_id + '.fits')
print hdulist.info()

import numpy as np
a = np.array(hdulist[1].data)
print "a.shape =", a.shape

print "Numpy: a.mean =", a.mean()

import scidbpy as sp
sdb = sp.connect('http://localhost:8080')

data_sdb = sdb.from_array(a.astype(np.float32))

res = data_sdb.mean()
print "SciDB: mean =", res[0]

print "Done!"
