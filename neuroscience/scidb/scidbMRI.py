import numpy as np
import matplotlib.pyplot as plt
import os
# %matplotlib inline
import botocore.session
import boto3
import dipy.core.gradients as dpg
import nibabel as nib

from dipy.segment.mask import median_otsu

import time
from scidbpy import connect

data_files = {'./bvals': '.../bvals',
              './bvecs': '.../bvecs',
              './data.nii.gz': '.../data.nii.gz'}


def load_data_from_aws():
    boto3.setup_default_session(profile_name='hcp')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('...')
    for k in data_files.keys():
        if not os.path.exists(k):
            print "%s does not exist - loading from aws" % k
            bucket.download_file(data_files[k], k)

    img = nib.load('./data.nii.gz')

    data = img.get_data()

    return data


def main():
    time_start = time.time()
    sdb = connect()

    print("Time passed: %.3fs" % (time.time() - time_start))
    data = load_data_from_aws()

    print "Datatype:", data.dtype

    if True:
        # Reduce data to work with for coding
        sh = data.shape
        data = data[int(sh[0] * .25): int(sh[0] * .75), 
                    int(sh[1] * .25): int(sh[1] * .75),
                    int(sh[2] * .25): int(sh[2] * .75)]

    print("Time passed: %.3fs" % (time.time() - time_start))
    
    gtab = dpg.gradient_table('./bvals', './bvecs', b0_threshold=10)

    print("Time passed: %.3fs" % (time.time() - time_start))

    data_sdb = sdb.from_array(data)

    # Creating mask
    raise()
    mean_b0 = data_sdb.compress(sdb.from_array(gtab.b0s_mask), axis=3)
    mean_b0 = mean_b0.mean(-1)
    _, mask = median_otsu(mean_b0.toarray(), 4, 2, False, vol_idx=np.where(gtab.b0s_mask), dilate=1)

    print("Time passed: %.3fs" % (time.time() - time_start))



if __name__ == "__main__":
    main()