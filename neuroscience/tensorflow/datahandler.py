import boto3
import nibabel as nib
import numpy as np
import os
try:
    import cPickle as pkl
except:
    import _pickle as pkl
import tensorflow as tf

# IDS
subjects = ["100307", "100408", "101006", "101107", "101309", "101410",
            "101915", "102311", "102816", "103111", "103515", "105014",
            "105115", "105216", "106016", "106319", "106521", "107321",
            "108121", "108323", "108525", "108828", "109123", "110411",
            "111312"]


def load_tf(id, shape):
    read = tf.read_file('./%d_data.npy' % id)
    arr = tf.decode_raw(read, out_type=tf.float32)
    arr = tf.reshape(arr, shape)
    return arr


def download(i_subject, download_only=False):

    subject_id = subjects[i_subject]

    print("downloading:", subject_id)

    session = boto3.session.Session()
    s3 = session.resource('s3')
    if not os.path.exists(subject_id + "_data.nii.gz"):
        s3.meta.client.download_file('imagedb-data', subject_id + "/data.nii.gz",
                                     subject_id + "_data.nii.gz")
    if not os.path.exists(subject_id + "_bvecs"):
        s3.meta.client.download_file('imagedb-data', subject_id + "/bvecs",
                                     subject_id + "_bvecs")
    if not os.path.exists(subject_id + "_bvals"):
        s3.meta.client.download_file('imagedb-data', subject_id + "/bvals",
                                     subject_id + "_bvals")

    print("downloaded", subject_id)

    if not download_only:
        datafile = subject_id + "_data.nii.gz"
        print("loading data file:", datafile)
        img = nib.load(datafile)
        data = img.get_data()
        return data, subject_id + "_bvals", subject_id + "_bvecs"


def read_ip_file(path):
    ip_ports = []
    ips = []
    port = 2500
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            ip_ports.append(line.strip() + ":" + str(port))
            ips.append(line.strip())
            port += 1

    return ip_ports, ips

