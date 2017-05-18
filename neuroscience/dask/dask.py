import boto3
from multiprocessing import Process, Array
import nibabel as nib
import numpy as np
import os.path as op
import sys, time, itertools, timeit

import dipy.core.gradients as dpg
import dipy.reconst.dti as dti
from dipy.segment.mask import median_otsu
from dipy.denoise import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

# use multi-processes
from dask import delayed, compute
import dask.multiprocessing 
from distributed import Executor
import distributed

subjects = [
"..."]


def download (id):
  print("downloading:", id)
  
  session = boto3.session.Session()
  s3 = session.resource('s3')
  s3.meta.client.download_file('...', id + "/data.nii.gz", id + "_data.nii.gz")
  s3.meta.client.download_file('...', id + "/bvecs", id + "_bvecs")
  s3.meta.client.download_file('...', id + "/bvals", id + "_bvals")
  
  print("downloaded", id)

  datafile = id + "_data.nii.gz"
  print("loading data file:", datafile)
  img = nib.load(datafile)
  data = img.get_data()
  affine = img.affine

  return data, data.shape, affine


def filter (id, images):

  print("filtering:", id)

  num_vols = images.shape[-1] # min 4

  bvalsFile = id + "_bvals"
  bvecsFile = id + "_bvecs"

  bvals = np.loadtxt(bvalsFile)
  bvecs = np.loadtxt(bvecsFile)

  # fix from Ariel
  gtab = dpg.gradient_table(bvals[:num_vols], bvecs[:, :num_vols], b0_threshold=10)
 
  filtered = images[..., gtab.b0s_mask]

  return (filtered, gtab)


# partition by voxels, groups of 10x10x10
def partition (data):

  [xp,yp,zp] = [10,10,10]
  [xSize,ySize,zSize] = [int(data.shape[0]/xp), int(data.shape[1]/yp), int(data.shape[2]/zp)] 

  slices = []
  for x,y,z in itertools.product(range(xp), range(yp), range(zp)):
    [xS, yS, zS] = [x*xSize, y*ySize, z*zSize]
    [xE, yE, zE] = [data.shape[0] if x == xp - 1 else (x+1)*xSize, \
                    data.shape[1] if y == yp - 1 else (y+1)*ySize, \
                    data.shape[2] if z == zp - 1 else (z+1)*zSize]
    
    if len(data.shape) == 4:
      slices.append(data[xS:xE, yS:yE, zS:zE, :])
    elif len(data.shape) == 3:
      slices.append(data[xS:xE, yS:yE, zS:zE])
    else: 
      raise Exception("unknown data size:", len(data.shape))

  print("id:", id, "num slices:", len(slices))
  return slices, len(slices)


def reassemble_means (means, shape):

  [xp,yp,zp] = [10,10,10]
  [xSize,ySize,zSize] = [int(shape[0]/xp), int(shape[1]/yp), int(shape[2]/zp)] 

  mean_b0 = np.empty(shape[:-1], dtype=np.float32)
    
  for (x,y,z),k in zip(itertools.product(range(xp), range(yp), range(zp)), \
                       range(len(means))):
    [xS, yS, zS] = [x*xSize, y*ySize, z*zSize]
    [xE, yE, zE] = [shape[0] if x == xp - 1 else (x+1)*xSize, \
                    shape[1] if y == yp - 1 else (y+1)*ySize, \
                    shape[2] if z == zp - 1 else (z+1)*zSize]
    #print("xs:", xS, "xe:", xE, "ys:", yS, "ye:", yE, "zs:", zS, "ze:", zE)

    mean_b0[xS:xE, yS:yE, zS:zE] = means[k] 

  return mean_b0


# old code
'''
def reassemble_model (fit_results, shape):

  [xp,yp,zp] = [10,10,10]
  [xSize,ySize,zSize] = [int(shape[0]/xp), int(shape[1]/yp), int(shape[2]/zp)] 

  fa = np.empty(shape[:-1], dtype=np.float32)
  md = np.empty(shape[:-1], dtype=np.float32)

  for (x,y,z),k in zip(itertools.product(range(xp), range(yp), range(zp)), \
                       range(len(fit_results))):
    [xS, yS, zS] = [x*xSize, y*ySize, z*zSize]
    [xE, yE, zE] = [shape[0] if x == xp - 1 else (x+1)*xSize, \
                    shape[1] if y == yp - 1 else (y+1)*ySize, \
                    shape[2] if z == zp - 1 else (z+1)*zSize]
 
    fa[xS:xE, yS:yE, zS:zE] = fit_results[k].fa
    md[xS:xE, yS:yE, zS:zE] = fit_results[k].md

  return (fa, md)
'''

def reassemble_model1 (fit_results, shape):

  [xp,yp,zp] = [10,10,10]
  [xSize,ySize,zSize] = [int(shape[0]/xp), int(shape[1]/yp), int(shape[2]/zp)] 

  r = np.empty(shape[:-1]) #, dtype=np.float32)

  for (x,y,z),k in zip(itertools.product(range(xp), range(yp), range(zp)), \
                       range(len(fit_results))):
    [xS, yS, zS] = [x*xSize, y*ySize, z*zSize]
    [xE, yE, zE] = [shape[0] if x == xp - 1 else (x+1)*xSize, \
                    shape[1] if y == yp - 1 else (y+1)*ySize, \
                    shape[2] if z == zp - 1 else (z+1)*zSize]
 
    r[xS:xE, yS:yE, zS:zE] = fit_results[k]

  return r


def reassemble_denoised (denoised_vols, shape):
  denoised = np.empty(shape, dtype=np.float32)
    
  for i in range(shape[-1]):
    denoised[..., i] = denoised_vols[i] 

  return denoised


def save_results (id, data, affine, filename):
  if data is not None:
    nib.save(nib.Nifti1Image(data, affine), filename)
    return filename
  else:
    return "none"


def print_sum(id, prefix, data):
  return "id: " + str(id) + " " + prefix + " sum: " + str(np.sum(data))


class Metadata:

  def __init__ (self):
    self.images = None
    self.images_shape = None
    self.affine = None
    self.slices = None
    self.num_slices = None
    self.mask = None
    self.gtab = None
    self.num_mask_slices = None
    self.denoised = None
    self.denoised_slices = None
    self.mask_slices = None


def run (ids, workers):

  start = time.time()
    
  for i in range(len(ids)):

    id = ids[i]

    #images, affine = download(id)
    #filtered, gtab = filter(id, images)
    
    #r = delayed(download)(id)
    #images, images_shape, affine = r[0], r[1], r[2]
    
    j = i % len(workers)
    r = e.submit(download, id, workers=[workers[j]])
    print("id:", id, "assigned to worker:", workers[j])
    import operator
    images, images_shape, affine = [e.submit(operator.getitem, r, i) for i in [0,1,2]]

    images = delayed(images)
    
    r = delayed(filter)(id, images)
    filtered, gtab = r[0], r[1]
 
    r = delayed(partition)(filtered)
    slices, num_slices = r[0], r[1]

    meta[id] = Metadata()
    meta[id].images = images
    meta[id].images_shape = images_shape
    meta[id].affine = affine
    meta[id].slices = slices
    meta[id].num_slices = num_slices
    meta[id].gtab = gtab


  ### BARRIER 1: evaluate number of slices and gtab
  for id in ids:    
    meta[id].num_slices = e.compute(meta[id].num_slices)
    meta[id].gtab = e.compute(meta[id].gtab)
    #meta[id].images = e.compute(meta[id].images)

  for id in ids:
    print("eval num_slices")
    meta[id].num_slices = meta[id].num_slices.result()    
    meta[id].gtab = meta[id].gtab.result()
    

    #sliced_means = [np.mean(s, -1) for s in slices]
    sliced_means = [delayed(np.mean)(meta[id].slices[i], -1) for i in range(meta[id].num_slices)]
   
    #mean = reassemble_means(sliced_means, images_shape)
    mean = delayed(reassemble_means)(sliced_means, images_shape)
 
    #_, mask = median_otsu(meta[id].mean, 4, 2, False, vol_idx=np.where(meta[id].gtab.b0s_mask), dilate=1)
    r = delayed(median_otsu)(mean, 4, 2, False, vol_idx=np.where(meta[id].gtab.b0s_mask), dilate=1)
    mask = r[1]
    meta[id].mask = mask
    # save mask if needed
    #nib.save(nib.Nifti1Image(mask.astype(int), meta[id].affine), id + "_mask.nii.gz")


  ### BARRIER 2: evaluate image shape for denoise partition
  for id in ids: 
    meta[id].images_shape = e.compute(meta[id].images_shape) 

  for id in ids:
    print("eval images_shape")
    meta[id].images_shape = meta[id].images_shape.result()


    num_vols = meta[id].images_shape[-1]
    vols = [meta[id].images[..., i] for i in range(num_vols)]
    
    #sigma_vols = [estimate_sigma(v) for v in vols]
    sigma_vols = [delayed(estimate_sigma)(v) for v in vols]
    #denoised_vols = [nlmeans.nlmeans(v, sigma=s, mask=mask) for v,s in zip(vols, sigma_vols)]
    denoised_vols = [delayed(nlmeans.nlmeans)(v, sigma=s, mask=meta[id].mask) for v,s in zip(vols, sigma_vols)]
   
    denoised = delayed(reassemble_denoised)(denoised_vols, meta[id].images_shape)
    
    r = delayed(partition)(meta[id].mask)
    mask_slices, num_mask_slices = r[0], r[1]
    
    r = delayed(partition)(denoised)
    denoised_slices, num_denoised_slices = r[0], r[1]
    
    meta[id].num_mask_slices = num_mask_slices
    meta[id].denoised_slices = denoised_slices
    meta[id].mask_slices = mask_slices

    ### debug
    #meta[id].denoised = denoised
    meta[id].denoised = None


  ### BARRIER 3: evaluate number of denoising partitions
  for id in ids:
    meta[id].num_mask_slices = e.compute(meta[id].num_mask_slices)

  L = [] # stores the final set of jobs to compute
  for id in ids:
    
    print("eval num_mask_slices")
    num_mask_slices = meta[id].num_mask_slices.result()

    ten_model = delayed(dti.TensorModel)(meta[id].gtab, min_signal=1)
   
    #mask_slices = partition(mask)
    #denoised_slices = partition(denoised)
    #ten_model = dti.TensorModel(meta[id].gtab, min_signal=1)

    #fit_slices = [ten_model.fit(d, m) for d,m in zip(meta[id].denoised_slices, meta[id].mask_slices)]
    fit_slices = [delayed(ten_model.fit)(meta[id].denoised_slices[i], meta[id].mask_slices[i])
                  for i in range(num_mask_slices)]
    
    #fa, md = reassemble_model(fit_slices, meta[id].images_shape)
    #r = delayed(reassemble_model)(fit_slices, meta[id].images_shape)
    #fa, md = r[0], r[1]

    fa = delayed(reassemble_model1)([fit_slices[i].fa for i in range(num_mask_slices)], meta[id].images_shape)
    md = delayed(reassemble_model1)([fit_slices[i].md for i in range(num_mask_slices)], meta[id].images_shape)
    meta[id].fa = fa
    meta[id].md = md
    

    # debug code below
    '''
    gtab = meta[id].gtab
    ten_model = dti.TensorModel(gtab, min_signal=1)
    
    images = e.compute(meta[id].images).result()
    mask = e.compute(meta[id].mask).result()
    ten_fit = ten_model.fit(images, mask=mask)
    fa = ten_fit.fa
    md = ten_fit.md
    '''

    ''' 
    mask = e.compute(meta[id].mask).result()
    denoised = e.compute(meta[id].denoised).result()

    mask_slices = partition(mask)
    denoised_slices = partition(denoised)
    ten_model = dti.TensorModel(e.compute(meta[id].gtab).result(), min_signal=1)

    fit_slices = [ten_model.fit(d, m) for d,m in zip(denoised_slices, mask_slices)]
    
    fa = reassemble_model1([fit_slices[i].fa for i in range(num_mask_slices)], meta[id].images_shape)
    md = reassemble_model1([fit_slices[i].md for i in range(num_mask_slices)], meta[id].images_shape)
    meta[id].fa = fa
    meta[id].md = md
    '''
  
    #L.append(delayed(np.sum)(meta[id].fa))
    #L.append(delayed(np.sum)(meta[id].md))
    L.append(delayed(print_sum)(id, "fa", meta[id].fa))
    L.append(delayed(print_sum)(id, "md", meta[id].md))

    # save images if needed
    #L.append(delayed(save_results)(id, meta[id].fa, meta[id].affine, "full_dask_" + id + "_dti_fa.nii.gz"))
    #L.append(delayed(save_results)(id, meta[id].md, meta[id].affine, "full_dask_" + id + "_dti_md.nii.gz"))  
    #L.append(delayed(save_results)(id, meta[id].denoised, meta[id].affine, "full_dask_" + id + "_denoised.nii.gz"))



  ### compute the final set up of jobs
  print("computing final set of jobs") 
  L = [e.compute(l) for l in L]
  print("waiting")
  L = [l.result() for l in L]
  print(L)
  #print("final:", e.has_what()) # shows which worker has what data

  print("time to run entire pipeline:", ((time.time() - start)))




if __name__ == "__main__":

  e = Executor("localhost:8786")

  meta = {}

  for i in [1,2,4,8,12,25]:
    e.restart()
    workers = list(e.has_what())
    print("workers:", workers)
    r = timeit.timeit("run(subjects[0:i], workers)",
                      "gc.enable(); from __main__ import run, subjects, i, meta, workers", number=1)
    print("# ids:", i, "time:", r)
