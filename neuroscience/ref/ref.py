import numpy as np
import os.path as op
import dipy.core.gradients as dpg
import nibabel as nib
import sys, time
import botocore.session
import boto3
from dipy.segment.mask import median_otsu



data_files = {'./bvals':'.../bvals',
               './bvecs':'.../bvecs',
               './data.nii.gz':'.../data.nii.gz'}


boto3.setup_default_session(profile_name='hcp')
s3 = boto3.resource('s3')
bucket = s3.Bucket('...')

for k in data_files.keys():
     if not op.exists(k):
         bucket.download_file(data_files[k], k)


#Load data
img = nib.load('./data.nii.gz')
data = img.get_data()

#build mask
gtab = dpg.gradient_table('./bvals', './bvecs', b0_threshold=10)
mean_b0 = np.mean(data[..., gtab.b0s_mask], -1)
_, mask = median_otsu(mean_b0, 4, 2, False, vol_idx=np.where(gtab.b0s_mask), dilate=1)

if not op.exists('./mask.nii.gz'):
     nib.save(nib.Nifti1Image(mask.astype(int), img.affine), 'mask.nii.gz')

# denoise
from dipy.denoise import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

sigma = estimate_sigma(data)
denoised_data = nlmeans.nlmeans(data, sigma=sigma, mask=mask)

if not op.exists('./denoised_data.nii.gz'):
    nib.save(nib.Nifti1Image(denoised_data, img.affine), 'denoised_data.nii.gz')

#tensor model
import dipy.reconst.dti as dti

ten_model = dti.TensorModel(gtab)
ten_fit = ten_model.fit(denoised_data, mask=mask)
nib.save(nib.Nifti1Image(ten_fit.fa, img.affine), 'dti_fa.nii.gz')
nib.save(nib.Nifti1Image(ten_fit.md, img.affine), 'dti_md.nii.gz')
