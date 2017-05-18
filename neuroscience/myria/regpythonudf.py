# create a connection.
from myria import *
import numpy
import json
from myria.connection import MyriaConnection
from myria.relation import MyriaRelation
from myria.udf import MyriaPythonFunction
from raco.types import STRING_TYPE, BOOLEAN_TYPE, LONG_TYPE, BLOB_TYPE

connection = MyriaConnection(rest_url='http://localhost:8753',execution_url='http://localhost:8080')


def mean(dt):
    return dt[0]/dt[1]

MyriaPythonFunction(mean, BLOB_TYPE).register()

def add(dt):
    tuplist =dt[1]
    retval= None
    for i in tuplist:
        if retval is None:
           retval = i[1]
        else:
           retval = retval+i[1]
    return retval

MyriaPythonFunction(add, BLOB_TYPE).register()


def medianO(dt):
    from dipy.segment.mask import median_otsu
    subID = dt[0]
    mean_b0 = dt[1]
    vols = [0,  16,  32,  48,  64,  80,  95, 112, 128, 144, 160, 176, 191,
       208, 224, 240, 256, 272]
    _, mask = median_otsu(mean_b0, 4, 2, False, vol_idx=vols, dilate=1)
    return mask

MyriaPythonFunction(medianO, BLOB_TYPE).register()


def fit_model(dt):
    import numpy as np
    import dipy.reconst.dti as dti
    import dipy.core.gradients as dpg
    tuplist = dt
    state = None
    for i in tuplist:
        imgid = i[1]
        flatmapid = i[2]
        img = np.asarray(i[3][0])
        shape = img.shape + (288,)
        if state is None:
            state = np.empty(shape)
            state[:,:,:,imgid]=img
        else:
            state[:,:,:,imgid]=img
    mask = i[3][1]
    gtab = dpg.gradient_table('/home/ubuntu/bvals', '/home/ubuntu/bvecs', b0_threshold=10)
    ten_model = dti.TensorModel(gtab, min_signal=1)
    ten_fit = ten_model.fit(state, mask=mask)
    return (flatmapid,ten_fit)

MyriaPythonFunction(fit_model, BLOB_TYPE).register()


def denoise(dt):
    from dipy.denoise import nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma
    import itertools
    item = dt[0]
    image = item[1]
    mask = item[2]
    sigma = estimate_sigma(image)
    denoised_data = nlmeans.nlmeans(image, sigma=sigma, mask=mask)
    [xp,yp,zp] = [4,4,4]
    [xSize,ySize,zSize] = [denoised_data.shape[0]/xp, denoised_data.shape[1]/yp, denoised_data.shape[2]/zp]
    datalist = []
    for x,y,z in itertools.product(range(xp), range(yp), range(zp)):
        [xS, yS, zS] = [x*xSize, y*ySize, z*zSize]
        [xE, yE, zE] = [denoised_data.shape[0] if x == xp - 1 else (x+1)*xSize, \
                        denoised_data.shape[1] if y == yp - 1 else (y+1)*ySize, \
                        denoised_data.shape[2] if z == zp - 1 else (z+1)*zSize]
        tup =(denoised_data[xS:xE, yS:yE, zS:zE],mask[xS:xE, yS:yE, zS:zE])
        datalist.append(tup)
    return datalist


MyriaPythonFunction(denoise, BLOB_TYPE,  multivalued=True ).register()

##recombine tm and save as nifti images
import nibabel as nib
import numpy as np
import itertools
import cPickle

img = nib.load('/home/ubuntu/100307_data.nii.gz')

shape = (145, 174, 145)
[xp,yp,zp] = [4,4,4]
[xSize,ySize,zSize] = [shape[0]/xp, shape[1]/yp,shape[2]/zp]

a = np.loadtxt('results.csv', dtype=str)

i = 0
arr =[]
for x,y,z in itertools.product(range(xp), range(yp), range(zp)):
    [xS,yS,zS] = [x*xSize,y*ySize,z*zSize]
    [xE, yE, zE] = [shape[0] if x==xp-1 else (x+1)*xSize,\
      shape[1] if y ==yp-1 else(y+1)*ySize,\
      shape[2] if z == zp-1 else(z+1)*zSize]
    arr.append((i,(xS,xE),(yS,yE),(zS,zE)))
    print arr[i]
    i=i+1

fa = np.empty(shape)
md = np.empty(shape)

for i in a:
    tm = cPickle.load(open(i,'rb'))
    ind = arr[tm[0]]
    fa[ind[1][0]:ind[1][1], ind[2][0]:ind[2][1],ind[3][0]:ind[3][1]] = tm[1].fa.T
    md[ind[1][0]:ind[1][1], ind[2][0]:ind[2][1],ind[3][0]:ind[3][1]] = tm[1].md.T

nib.save(nib.Nifti1Image(fa, img.affine), "my_100307_4.fa.nii.gz")
nib.save(nib.Nifti1Image(md, img.affine), "my_100307_4.md.nii.gz")
