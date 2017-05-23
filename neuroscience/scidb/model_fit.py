import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import nibabel as nib
import os
import re
import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu
from dipy.reconst import dti

from scidbpy import connect
from scidbpy import robust
from scidbpy import save

def pinv(sdb, a, rcond=1e-15):
    # a, wrap = _makearray(a)

    # Don't know if ignoring the conjugate leeds to problems...

    print a.shape

    a = sdb.from_array(a.toarray().conjugate())
    s = robust.gesvd(a, "'S'")
    vt = robust.gesvd(a, "'VT'")
    u = robust.gesvd(a, "'U'")
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond*s.max(0)

    new_s = np.zeros(min(n, m))
    for i in range(min(n, m)):
        if s[i] > cutoff:
            new_s[i] = 1./s[i]

    s = sdb.from_array(new_s)
    res = sdb.dot(vt.T, (u * s.T).T)
    return res


def dipys_3d_pinv(sdb, a, rcond=1e-15):
    # a, wrap = _makearray(a)

    # Don't know if ignoring the conjugate leeds to problems...

    # a = a.conjugate()


    swap = np.arange(len(a.shape))
    swap[[-2, -1]] = swap[[-1, -2]]

    s = robust.gesvd(a, "'S'")
    vt = robust.gesvd(a, "'VT'")
    u = robust.gesvd(a, "'U'")

    m = s.max(-1).reshape(s.shape[0], 1)
    cutoff = rcond*m

    mask = s.toarray() > cutoff.toarray()
    s[mask] = 1. / s[mask]
    s[~mask] = 0

    return np.einsum('...ij,...jk',
                     np.transpose(v, swap) * s[..., None, :],
                     np.transpose(u, swap))



def convert_array_type(sdb, arr, to_type):
    s = arr.schema
    from_type = re.findall(r":[\w\d]+", s)[0][1:]
    type_index = s.find(from_type)
    return sdb.afl.cast(arr, s[:type_index] + 
                             to_type.decode("unicode-escape") + 
                             s[type_index + len(from_type):])


home_path = os.path.expanduser("~")

sdb = connect('http://localhost:8080')

#Firstly, we need to make the following SciDB-py filter work correctly
x = np.random.randint(low=1, high=10, size=(5, 4))
x_sdb = sdb.from_array(x)
print 'x_sdb =\n', x_sdb.toarray()
large5 = sdb.afl.filter(x_sdb, x_sdb.attribute(0)>5)
print 'large5 =\n', large5.toarray()
inverse_sdb = 1. / x_sdb
print 'inverse_sdb =\n', inverse_sdb.toarray() 

print "Start loading data from disk: ", str(time.ctime())
img = nib.load('/home/ubuntu/download/data.nii.gz')
data = img.get_data()

print "Start loading data to scidb: ", str(time.ctime())

#For an entire subject (we need to read in two halves to avoid Shim crash):
#data_sdb_p0 = sdb.from_array(data[0:72,:,:,:])
#data_sdb_p1 = sdb.from_array(data[72:145,:,:,:])
#data_sdb = sdb.concatenate((data_sdb_p0, data_sdb_p1), axis=0)

#Use a small set for quick testing
sh = data.shape
data_sdb = sdb.from_array(data[sh[0]/2-1: sh[0]/2+1,
			       sh[1]/2-1: sh[1]/2+1,
			       sh[2]/2-1: sh[2]/2+1,
			       :16]) 

print "Ingestion done: ", str(time.ctime())
print data_sdb.shape

#
# Read bvals and bvecs; needs to be done for the mask building anyways and will be 
# needed for the design matrix 
# 

gtab = dpg.gradient_table(home_path + '/download/bvals', 
				          home_path + '/download/bvecs', 
				          b0_threshold=10)

# Truncate to a size that fits the small data:

gtab.bvecs = gtab.bvecs[:16]
gtab.bvals = gtab.bvals[:16]
gtab.gradients = gtab.gradients[:16]

#
#Now we want to reimplement in SciDB this one: https://github.com/nipy/dipy/blob/master/dipy/reconst/dti.py: TensorModel.fit
#

#Reshape the data: data_in_mask = np.reshape(data, (-1, data.shape[-1]))
new_shape = (data_sdb.shape[-1],) + data_sdb.shape[:-1]
data_in_mask = data_sdb.reshape(new_shape)
print "Reshaping done: ", str(time.ctime())
print data_in_mask.shape

"""

#Find the minimum positive value: min_signal = _min_positive_signal(data)
data_pos = sdb.afl.filter(data_in_mask, data_in_mask.attribute[0]>0)
min_signal = data_pos.min()
print "Minimum positive value done: ", str(time.ctime())
print min_signal


#Filter out values larger than min_pos: data_in_mask = np.maximum(data_in_mask, min_signal)
data_neg = sdb.filter(data_in_mask, data_in_mask.attribute[0]<=0)
data_fill = sdb.substitute(data_neg, min_signal)
data_in_mask = sdb.filter(data_fill, data_fill.attribute[0]>0)
print "Truncation done: ", str(time.ctime())
print data_in_mask.shape

"""

#TODO: params_in_mask = self.fit_method(self.design_matrix, data_in_mask, *self.args, **self.kwargs), where we need to re-implement
#       def wls_fit_tensor(design_matrix, data). This is a fancy math function including an Einstein Summation...

data_in_mask = data_sdb.reshape((-1, data_sdb.shape[-1]))
data_in_mask = convert_array_type(sdb, data_in_mask, "double")

# Create design matrix B

design_matrix = sdb.from_array(dti.design_matrix(gtab))

# wls fit 
# apparantly one has to run the svd three times to get all three values...

u = robust.gesvd(design_matrix, "'U'")

ols_fit = sdb.dot(u, u.T)

log_s = sdb.log(data_in_mask)

# The einsum: w = np.exp(np.einsum('...ij,...j', ols_fit, log_s))

w = sdb.exp(sdb.dot(ols_fit, log_s.T).T)

# p = sdb.from_array(dipys_3d_pinv(sdb, sdb.dstack([design_matrix[:, i] * w for i in range(design_matrix.shape[1])])))

pinv_arg = sdb.dstack([design_matrix[:, i] * w for i in range(design_matrix.shape[1])])
p = sdb.dstack([pinv(sdb, pinv_arg[i]) for i in range(pinv_arg.shape[0])]).transpose((2, 0, 1))

# p = dipys_3d_pinv(sdb, pinv_arg)
print p.shape, log_s.shape

r_arg = w * log_s
r = sdb.dot(p, r_arg)

# Now: eig_from_lo_tri()

_lt_indices = np.array([[0, 1, 3],
                        [1, 2, 4],
                        [3, 4, 5]])

# Fall back to python for now:

r = sdb.from_array(r.toarray()[..., _lt_indices])

min_diffusivity = 1e-6 / -sdb.min(design_matrix)

