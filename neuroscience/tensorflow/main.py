import datahandler as dh
import distributer as dt
import numpy as np
import sys
import tensorflow as tf
import time

from dipy.segment.mask import median_otsu
import dipy.core.gradients as dpg


def end_to_end_iteration(sess, tf_cluster, data_ids):
    time_download_start = time.time()
    datasets = []
    gtabs = []
    for data_id in data_ids:
        data, bvals_path, bvecs_path = dh.download(data_id)
        datasets.append(data)
        gtabs.append(dpg.gradient_table(bvals_path, bvecs_path,
                                        b0_threshold=10))
    print "Time downloading %.3f" % (time.time() - time_download_start)

    time_filter_start = time.time()
    print("\nImage filtering\n")
    filtered_datasets = dt.parallel_image_filtering(sess, tf_cluster,
                                                    datasets, gtabs,
                                                    n_parts=288)
    print "Time filtering: %.3fs" % (time.time() - time_filter_start)

    time_mean_start = time.time()
    print("\nMean\n")
    mean_data = dt.parallel_mean(sess, tf_cluster, filtered_datasets,
                                 len(data_ids), n_parts=0)
    print "Time mean: %.3fs" % (time.time() - time_mean_start)

    filtered_datasets = None

    time_median_start = time.time()
    print("\nMedian Otsu\n")
    masks = dt.parallel_median_otsu(sess, tf_cluster, mean_data)
    print "Time median otsu: %.3fs" % (time.time() - time_median_start)

    mean_data = None


    time_denoise_start = time.time()
    print("\nDenoise\n")
    datasets = dt.parallel_denoise(sess, tf_cluster, datasets, masks,
                                   depth=1)
    print "Time denoising: %.3fs" % (time.time() - time_denoise_start)

    print("Start measurement")
    time_model_start = time.time()
    models = dt.parallel_modelbuilding(sess, tf_cluster, masks,
                                       datasets, gtabs, n_parts=512)
    print("Time model: %.3fs" % (time.time() - time_model_start))

    datasets = None
    models = None
    masks = None
    gtabs = None


def end_to_end(n_datasets, ip_file_path, stride=8):
    time_total_start = time.time()

    data_ids = range(n_datasets)

    ip_ports, ips = dh.read_ip_file(ip_file_path)
    tf_cluster = dt.TfCluster(ip_ports[:])

    for data_ids_block in [data_ids[i: i+stride]
                           for i in range(0, len(data_ids), stride)]:
        with tf.Session("grpc://%s" % tf_cluster.host) as sess:
            time_iteration = time.time()
            end_to_end_iteration(sess, tf_cluster, data_ids_block)
            print("Time iteration: %.3fs" % (time.time() - time_iteration))
        tf.reset_default_graph()

    print "Time overall: %.3fs" % (time.time() - time_total_start)


def measure_mask(n_datasets, ip_file_path):
    data_ids = range(n_datasets)

    ip_ports, ips = dh.read_ip_file(ip_file_path)

    tf_cluster = dt.TfCluster(ip_ports[:])

    filtered_data = []
    for data_id in data_ids:
        data, bvals_path, bvecs_path = dh.download(data_id)
        gtab = dpg.gradient_table(bvals_path, bvecs_path,
                                  b0_threshold=10)
        filtered_data.append(data[..., gtab.b0s_mask])

    with tf.Session("grpc://%s" % tf_cluster.host) as sess:
        print("\nMean\n")
        dt.parallel_mean(sess, tf_cluster, filtered_data)

def measure_mean(n_real_datasets, n_datasets, ip_file_path, n_parts=1):
    data_ids = range(n_real_datasets)

    ip_ports, ips = dh.read_ip_file(ip_file_path)

    tf_cluster = dt.TfCluster(ip_ports[:])

    filtered_data = []
    for data_id in data_ids:
        data, bvals_path, bvecs_path = dh.download(data_id)

        gtab = dpg.gradient_table(bvals_path, bvecs_path,
                                  b0_threshold=10)
        filtered_data.append(data[..., gtab.b0s_mask])
        print(filtered_data[-1].shape)

    with tf.Session("grpc://%s" % tf_cluster.host) as sess:
        print("\nMean\n")
        dt.parallel_mean(sess, tf_cluster, filtered_data, n_datasets, n_parts=n_parts)


def measure_image_filtering(n_datasets, multiplicator, ip_file_path):
    data_ids = range(n_datasets)

    ip_ports, ips = dh.read_ip_file(ip_file_path)
    tf_cluster = dt.TfCluster(ip_ports)

    datasets = []
    path_sets = []
    for data_id in data_ids:
        data, bvals_path, bvecs_path = dh.download(data_id)
        datasets.append(data)
        print(datasets[-1].shape)
        path_sets.append([bvals_path, bvecs_path])

    with tf.Session("grpc://%s" % tf_cluster.host) as sess:
        print("\nImage filtering\n")
        dt.parallel_image_filtering(sess, tf_cluster, datasets, path_sets,
                                    multiplicator, n_parts=288)


def measure_median_filter(n_datasets, ip_file_path):
    data_ids = range(n_datasets)

    ip_ports, ips = dh.read_ip_file(ip_file_path)

    tf_cluster = dt.TfCluster(ip_ports[:])

    mean_data = []
    for data_id in data_ids:
        data, bvals_path, bvecs_path = dh.download(data_id)
        gtab = dpg.gradient_table(bvals_path, bvecs_path,
                                  b0_threshold=10)
        mean_data.append(np.mean(data[..., gtab.b0s_mask], -1))

    with tf.Session("grpc://%s" % tf_cluster.host) as sess:
        print("\nMedian\n")
        dt.parallel_median_otsu(sess, tf_cluster, mean_data)


def measure_denoising(n_datasets, ip_file_path):
    data_ids = range(n_datasets)

    ip_ports, ips = dh.read_ip_file(ip_file_path)

    tf_cluster = dt.TfCluster(ip_ports[:])

    datasets = []
    masks = []
    for data_id in data_ids:
        data, bvals_path, bvecs_path = dh.download(data_id)

        datasets.append(data)
        gtab = dpg.gradient_table(bvals_path, bvecs_path, b0_threshold=10)
        mean_b0 = np.mean(data[..., gtab.b0s_mask], -1)
        _, mask = median_otsu(mean_b0, 4, 2, False,
                              vol_idx=np.where(gtab.b0s_mask), dilate=1)
        masks.append(mask)

    with tf.Session("grpc://%s" % tf_cluster.host) as sess:
        dt.parallel_denoise(sess, tf_cluster, datasets, masks, depth=1)


def measure_model_building(n_datasets, ip_file_path, stride=9):
    all_data_ids = range(n_datasets)

    for data_ids in [all_data_ids[i:i + stride]
                     for i in range(0, len(all_data_ids), stride)]:

        ip_ports, ips = dh.read_ip_file(ip_file_path)

        tf_cluster = dt.TfCluster(ip_ports[:])

        datasets = []
        masks = []
        gtabs = []
        for data_id in data_ids:
            data, bvals_path, bvecs_path = dh.download(data_id)

            datasets.append(data)
            gtab = dpg.gradient_table(bvals_path, bvecs_path, b0_threshold=10)
            mean_b0 = np.mean(data[..., gtab.b0s_mask], -1)
            _, mask = median_otsu(mean_b0, 4, 2, False,
                                  vol_idx=np.where(gtab.b0s_mask), dilate=1)
            masks.append(mask)
            gtabs.append(gtab)

        print("Start measurement")
        time_start = time.time()
        with tf.Session("grpc://%s" % tf_cluster.host) as sess:
            dt.parallel_modelbuilding(sess, tf_cluster, masks, datasets, gtabs,
                                      n_parts=64)
        print("Time iteration: %.3fs" % (time.time()-time_start))


def measure_downloading(n_datasets_list):
    times = []
    time_start = time.time()

    ids = []
    for data_id in range(np.max(n_datasets_list)):
        data = dh.download(data_id, download_only=False)
        times.append(time.time()-time_start)
        ids.append(data_id)

    for i_n in range(len(ids)):
        print("Time for %d datasets: %.3fs" % (ids[i_n], times[i_n]))


if __name__ == "__main__":
    end_to_end(25, "ips", 2)
