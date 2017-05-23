#####
# Author: DFZ (dzhao@cs.washington.edu)
# Description: Modified from the workflow from UW DB and eScience
# History
#    5/9/2016: add time stamps; remove auto download of input data; fix various librari errors
#####


# coding: utf-8

# ## An end-to-end pipeline for the analysis of HCP diffusion MRI data using Dipy

# MRI benchmark end-toend as described works on a numpy array extrated from the NifTi  image format. The canonical use case  includes three steps:
# 1. Load Data
# 2. Clean Data
# 3. Create a Model
#
#
# ### Step I: Load Data.
# Data used for this benchmark is from the Human Connectome Project. This data lives in S3.  The data is downloaded as  NifTi image and then extracted to a four dimensional Numpy Array (x, y, z, Val). To calibirate the image data, additional data in the form  of the b-values and b-vectors is provided. B-vector is a three dimensional array of unit vectors and b-vals are a 1-D array indicating the strength of the MR signal.
# Next couple  sections walk through loading this data.
#

# In[15]:
import time
from dipy import denoise
print "Start at " + str(time.ctime())

import numpy as np
#import matplotlib.pyplot as plt
import os.path as op
#get_ipython().magic(u'matplotlib inline')


# #### Get the data from AWS S3
# Credentials for acquiring the data must live in '.aws/credentials', in a [HCP] section:
#
#     [hcp]
#     AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXX
#     AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXX
# with the credentials you got [from HCP](https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS)

# In[16]:

#data_files = {'./bvals':'HCP/994273/T1w/Diffusion/bvals',
#              './bvecs':'HCP/994273/T1w/Diffusion/bvecs',
#              './data.nii.gz':'HCP/994273/T1w/Diffusion/data.nii.gz'}
#read data.
#import botocore.session
#import boto3

#boto3.setup_default_session(profile_name='hcp')
#s3 = boto3.resource('s3')
#bucket = s3.Bucket('hcp-openaccess')

#for k in data_files.keys():
#    if not op.exists(k):
#        bucket.download_file(data_files[k], k)


# #### Data-description:
# data.nii.gz  is a series of diffusion MRI files in NIfTI format for one subject. bvec is the direction of the diffusion vector and bval is the value corresponding to that direction, provided by the MRI equipment/settings. These three files together  constitute the dataset  required for computations below.
# The sample data has the following dimensions ( 145,174,145,288). The bval-bvec similarly have length of 288.
#

# In[17]:

# load data:
import dipy.core.gradients as dpg
import nibabel as nib

DATA_LOC = "/home/ubuntu/mri_data/101107/"

img = nib.load(DATA_LOC + 'data.nii.gz')
data = img.get_data()


# In[ ]:


#print ./bvals

print " The image data is a "
print data.shape
print "vector, with first three representing the voxel location."
#NIfTI images have an affine relating the voxel coordinates to world coordinates in RAS+ space
print "image affine, ",img.affine


# ### Step 2: Cleaning Data
# A large part of this data is not needed for the computation. In this section, data from the previous section is cleaned up before it can be used for modelling.
#
# 1. Settign threshold on the  gradient table, Te gradient table is ocnstrucuted rom bvec/bvals. Using a threshold we only consider a subset of the values, where the signal/noise ration is high. For this specific example, a threshold of b0 values <=10, is used to mask out any of images with low signal to noise ratio. After this step, only 18 of 288 values are considered, reducing the (145,174,145,288) vector to ( 145,174,145,18)
#
# 2. Brain Extraction: Brain extraction is the process of segmenting brain from non-brain tissues such as skull, scalp, eyes, or neck in whole-head MR images and without removing any part of the brain. This benchmark uses a median filter smoothing of the input_volumes vol_idx and an automatic histogram Otsu thresholding technique.
# Mean value for each x,y,z location is first calculated(this only includes the 18 readings for each location selected from the previous step). Next median_ostu will be used to convert this grascale images to a  binary mask. Hereafter this mask is used to  limit the locations for which the computation is run in the follwoing steps.
#
# 3. Denoising with NL means: Last step of the cleaning process uses anon-local means filter [Coupe2008] to calulate a denoised voxel for each location. The denoised voxel is computed as the weighted  sum of all the neighboring voxels.
#
# (http://nipy.org/dipy/examples_built/denoise_nlmeans.html#example-denoise-nlmeans)

# In[4]:

import dipy.core.gradients as dpg
import nibabel as nib
# Read in the gradient table, setting the threshold to 10, so that low values are considered as 0:
gtab = dpg.gradient_table(DATA_LOC + 'bvals', DATA_LOC + 'bvecs', b0_threshold=10)

print 'gtab.b0s_mask.shape = ', gtab.b0s_mask.shape

# We look at the average non diffusion-weighted as data for brain extraction
mean_b0 = np.mean(data[..., gtab.b0s_mask], -1)

if not op.exists(DATA_LOC + 'mask.nii.gz'):
    from dipy.segment.mask import median_otsu
    _, mask = median_otsu(mean_b0, 4, 2, False, vol_idx=np.where(gtab.b0s_mask), dilate=1)
    nib.save(nib.Nifti1Image(mask.astype(int), img.affine), DATA_LOC + 'mask.nii.gz')
else:
    mask = nib.load(DATA_LOC + 'mask.nii.gz').get_data()

#exit(0) #so, this stops here to just complete computing the mask

if not op.exists(DATA_LOC + 'denoised_data.nii.gz'):
    from dipy.denoise import nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma
    sigma = estimate_sigma(data)
    denoised_data = nlmeans.nlmeans(data, sigma=sigma, mask=mask)
    nib.save(nib.Nifti1Image(denoised_data, img.affine), DATA_LOC + 'denoised_data.nii.gz')
else:
    denoised_data = nib.load(DATA_LOC + 'denoised_data.nii.gz').get_data()


# ###  Step 3: Building a Tensor Model.
# The Denoised data is  then used to build a Diffusion tensor model. This model describes diffusion within a voxel.
#
# $$\frac{S(\mathbf{g}, b)}{S_0} = e^{-b\mathbf{g}^T \mathbf{D} \mathbf{g}}$$
# Where \mathbf{g} is a unit vector in 3 space indicating the direction of measurement and b are the parameters of measurement, such as the strength and duration of diffusion-weighting gradient. $$S(\mathbf{g}, b)$$ is the diffusion-weighted signal measured and S_0 is the signal conducted in a measurement with no diffusion weighting. $$\mathbf{D}$$ is a positive-definite quadratic form, which contains six free parameters to be fit. These six parameters are:
#
# $$\mathbf{D} = \begin{pmatrix} D_{xx} & D_{xy} & D_{xz} \\
#                     D_{yx} & D_{yy} & D_{yz} \\
#                     D_{zx} & D_{zy} & D_{zz} \\ \end{pmatrix}$$
#
#  The parameters for the tensor model are initialized from teh bvals/bvec data. From this a tensor model is fitted for each voxel, which describes diffusion for each voxel.
#
# The fit method creates a TensorFit object which contains the fitting parameters and other attributes of the model. For example we can generate fractional anisotropy (FA) from the eigen-values of the tensor. FA is used to characterize the degree to which the distribution of diffusion in a voxel is directional. That is, whether there is relatively unrestricted diffusion in one particular direction.
#
# Mathematically, FA is defined as the normalized variance of the eigen-values of the tensor:
#
# $$FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
#             \lambda_3)^2+(\lambda_2-\lambda_3)^2}{\lambda_1^2+
#             \lambda_2^2+\lambda_3^2}}$$
#
# #### Segmentation
# In the last step of the benchmark, we take the Tensor model to do a coarse grained classifier to classify the white  and gray matter in the brain.
# FA is a measure of the anisotropy of diffusion in each voxel in the image. Because white matter tends to have higher anisotropy, we can use it as a way to roughly segment the image into portions that contain white matter and the non-white matter portion of the image. This allows us to reduce the computational demand of subsequent steps by only performing them in the masked regions of the image.
#
# See [example and explanations](http://nipy.org/dipy/examples_built/reconst_dti.html#example-reconst-dti)

# In[23]:

import dipy.reconst.dti as dti

ten_model = dti.TensorModel(gtab)
print("DFZ: denoised_data.shape = " + str(denoised_data.shape))
ten_fit = ten_model.fit(denoised_data[0:29,0:87,0:29], mask=mask[0:29,0:87,0:29]) #DFZ: isn't this denoised_data?

nib.save(nib.Nifti1Image(ten_fit.fa, img.affine), DATA_LOC + 'dti_fa.nii.gz')
nib.save(nib.Nifti1Image(ten_fit.md, img.affine), DATA_LOC + 'dti_md.nii.gz')

#create a tensor model from the MRI machine based gradient table and  fitting the 'non-denoised' data
# to the tensor model, given the mask.


# In[ ]:

#DFZ: fa -> ten_fit.fa?
wm_mask = np.zeros(ten_fit.fa.shape, dtype=bool)
wm_mask[ten_fit.fa>0.2] = 1

print "End at " + str(time.ctime())
#
