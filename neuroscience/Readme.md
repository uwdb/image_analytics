
# Data
Data for the neuroscience use case is available at [HCP](https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS)


## Get the data from AWS S3
We assume that you have a file '.aws/credentials', 
that includes a section with credentials needed to access HCP data.
```
[hcp]
AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXX
```

Following code excerpt shows how to access the data from the hcp-openaccess s3 bucket.
```python
import botocore.session
import boto3
boto3.setup_default_session(profile_name='hcp')
s3 = boto3.resource('s3')
bucket = s3.Bucket('hcp-openaccess')

 data_files = {'./bvals':'HCP/994273/T1w/Diffusion/bvals', 
              './bvecs':'HCP/994273/T1w/Diffusion/bvecs', 
              './data.nii.gz':'HCP/994273/T1w/Diffusion/data.nii.gz'}
         
for k in data_files.keys():
    if not op.exists(k):
        bucket.download_file(data_files[k], k)
```
