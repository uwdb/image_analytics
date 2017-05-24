import denoise as dn
import dipy.reconst.dti as dti
import mask as ma
import numpy as np
import tensorflow as tf
import time

import model_building as mb


def parallel_mean(sess, tf_cluster, arrs, n_datasets):
    shs = [arrs[i].shape for i in range(n_datasets)]

    zs = range(len(arrs))

    waves = tf_cluster.partition_work_waves(len(zs), use_host=False)

    pl_inputs = []
    work = []
    for i_worker in range(len(waves[0])):
        with tf.device(waves[0][i_worker]):
            print(waves[0][i_worker])
            pl_inputs.append(tf.placeholder(tf.float32,
                                            shape=shs[i_worker],
                                            name="raw_data_block_%d" %
                                                 i_worker))

            work.append(ma.mean_c(pl_inputs[-1]))

    mean_data = []
    time_inits = []
    for i_wave in range(len(waves)):
        time_start = time.time()

        this_zs = zs[i_wave * len(waves[0]):
                     i_wave * len(waves[0]) + len(waves[i_wave])]

        zip_d = zip(pl_inputs[:len(waves[i_wave])],
                    [arrs[z % len(arrs)] for z in this_zs])

        time_inits.append(time.time()-time_start)

        mean_data += sess.run(work[:len(waves[i_wave])],
                              feed_dict={i: d for i, d in zip_d})

    mean_data = [m.astype(np.bool) for m in mean_data]
    return mean_data


def parallel_image_filtering(sess, tf_cluster, arrs, gtabs, n_parts=288):
    print(arrs[0].shape[-1] % n_parts)
    assert arrs[0].shape[-1] % n_parts == 0

    sh = list(arrs[0].shape[:-1]) + [arrs[0].shape[-1]/n_parts]

    pl_inputs = []
    gtab_inputs = []
    work = []

    zs = [[int(np.floor(float(i)/n_parts)), i % n_parts]
          for i in range(len(arrs) * n_parts)]
    waves = tf_cluster.partition_work_waves(len(zs), use_host=False)

    for i_worker in range(len(waves[0])):
        with tf.device(waves[0][i_worker]):
            pl_inputs.append(tf.placeholder(tf.float32,
                                            shape=sh,
                                            name="raw_data_block_%d" %
                                                 i_worker))
            gtab_inputs.append(tf.placeholder(tf.bool,
                                              shape=[sh[-1]],
                                              name="gtab_block_%d" %
                                                   i_worker))

            work.append(ma.image_filtering(pl_inputs[-1], gtab_inputs[-1]))

    results = []
    for i_wave in range(len(waves)):
        this_zs = zs[i_wave * len(waves[0]):
                     i_wave * len(waves[0]) + len(waves[i_wave])]

        feed_dict = {a: d for a, d in
                     zip(pl_inputs[:len(waves[i_wave])],
                         [arrs[this_zs[j][0] % len(arrs)][..., sh[-1] * this_zs[j][1]: sh[-1]*(this_zs[j][1]+1)]
                          for j in range(len(this_zs))])}
        feed_dict.update({a: d for a, d in
                          zip(gtab_inputs[:len(waves[i_wave])],
                              [gtabs[this_zs[j][0] % len(arrs)].b0s_mask[..., sh[-1]*this_zs[j][1]: sh[-1]*(this_zs[j][1]+1)]
                               for j in range(len(this_zs))])})

        results += sess.run(work[:len(waves[i_wave])], feed_dict=feed_dict)

    filtered_arrs = []
    for i_arr in range(len(arrs)):
        filtered_arrs.append(np.concatenate(
            [x for x in results[i_arr * 288: (i_arr+1) * 288]
             if not x.shape[-1] == 0], axis=-1).astype(np.float32))
    return filtered_arrs


def parallel_median_otsu(sess, tf_cluster, arrs):
    masks = []
    for arr in arrs:
        mask = ma.multi_median(arr, 4, 2)
        mask = mask >= ma.otsu(mask)
        masks.append(mask)

    sh = list(masks[0].shape)
    print(sh)

    pl_inputs = []
    work = []

    zs = range(len(masks))
    waves = tf_cluster.partition_work_waves(len(zs), use_host=False)

    for i_worker in range(len(waves[0])):
        with tf.device(waves[0][i_worker]):
            pl_inputs.append(tf.placeholder(tf.float32,
                                            shape=sh,
                                            name="mean_block_%d" %
                                                 i_worker))

            work.append(ma.dilation(pl_inputs[-1]))

    d_masks = []
    for i_wave in range(len(waves)):
        this_zs = zs[i_wave * len(waves[0]):
                     i_wave * len(waves[0]) + len(waves[i_wave])]

        z_data = zip(pl_inputs[:len(waves[i_wave])],
                     [masks[z] for z in this_zs])

        d_masks += sess.run(work[:len(waves[i_wave])],
                            feed_dict={i: d for i, d in z_data})

    for i_mask in range(len(d_masks)):
        d_masks[i_mask] = d_masks[i_mask][..., 0]

    return d_masks


def parallel_denoise(sess, tf_cluster, arrs, masks, depth=1):
    assert len(arrs) == len(masks)
    assert arrs[0].shape[-1] % depth == 0

    sh = list(arrs[0].shape[:-1]) + [depth]

    pl_inputs = []
    work = []

    zs = range(0, arrs[0].shape[-1], depth)
    waves = tf_cluster.partition_work_waves(len(zs), use_host=False)

    for i_worker in range(len(waves[0])):
        with tf.device(waves[0][i_worker]):
            pl_inputs.append(tf.placeholder(tf.float32,
                                            shape=sh,
                                            name="raw_data_block_%d" %
                                                 i_worker))

            work.append(dn.nlmeans(pl_inputs[-1],
                                   dn.estimate_sigma(pl_inputs[-1], depth), sh,
                                   depth))

    denoised_arrs = []
    for i_arr in range(len(arrs)):
        arrs[0] *= masks[i_arr][..., None]
        denoised_arr = []
        for i_wave in range(len(waves)):
            this_zs = zs[i_wave*len(waves[0]): i_wave*len(waves[0])+len(waves[i_wave])]
            denoised_arr += sess.run(work[:len(waves[i_wave])],
                                     feed_dict={i: d for i, d in
                                                zip(pl_inputs[: len(waves[i_wave])],
                                                    [arrs[0][..., z: z + depth]
                                                     for z in this_zs])})
        del arrs[0]
        denoised_arrs.append(np.concatenate(denoised_arr, axis=-1))

    return denoised_arrs


def parallel_modelbuilding(sess, tf_cluster, masks, datasets, gtabs, n_parts=16):

    ten_model = dti.TensorModel(gtabs[0])
    design_matrix = ten_model.design_matrix

    nonzero_indices_list = [np.nonzero(masks[i]) for i in range(len(masks))]

    max_length = np.max([len(nonzero_indices_list[i][0])
                         for i in range(len(nonzero_indices_list))])

    stride = int(np.ceil(max_length / float(n_parts)))

    dim_sh = [stride, datasets[0].shape[-1]]

    waves = tf_cluster.partition_work_waves(
        int(np.ceil(max_length / stride)), use_host=False)

    dim_inputs = []
    cnt_inputs = []
    work = []
    dm_input = tf.placeholder(tf.float64, shape=design_matrix.shape,
                              name="dm")
    for i_worker in range(len(waves[0])):
        with tf.device(waves[0][i_worker]):
            dim_inputs.append(tf.placeholder(tf.float64,
                                             shape=dim_sh,
                                             name="dim_%d" % i_worker))

            cnt_inputs.append(tf.placeholder(tf.int32,
                                             shape=1,
                                             name="counter_%d" % i_worker))

            work.append(mb.model_building(dim_inputs[-1], dm_input,
                                          cnt_inputs[-1]))

    fas = []
    for i_data in range(len(datasets)):
        ten_model = dti.TensorModel(gtabs[i_data])
        design_matrix = ten_model.design_matrix
        nonzero_indices = nonzero_indices_list[i_data]

        dti_params = np.zeros(datasets[i_data].shape[:-1] + (12,))
        cnt = 1
        thread_mask = np.zeros(masks[i_data].shape, dtype=np.int)
        data = datasets[i_data]
        waves = tf_cluster.partition_work_waves(
            int(np.ceil(len(nonzero_indices[0]) / stride)), use_host=False)

        for i_wave in range(len(waves)):
            counter = []
            data_in_mask_list = []
            for i_worker in range(len(waves[0])):
                step = (cnt-1) * stride
                thread_mask[nonzero_indices[0][step: step + stride],
                            nonzero_indices[1][step: step + stride],
                            nonzero_indices[2][step: step + stride]] = cnt

                data_in_mask = \
                    np.reshape(data[nonzero_indices[0][step: step + stride],
                                    nonzero_indices[1][step: step + stride],
                                    nonzero_indices[2][step: step + stride]],
                               (-1, data.shape[-1]))
                data_in_mask = np.maximum(data_in_mask, 0.0001)

                data_in_mask_list.append(data_in_mask)
                counter.append(np.array([cnt]))
                cnt += 1

            feed_dict = {i: d for i, d in zip(dim_inputs[: len(waves[i_wave])],
                                              data_in_mask_list)}

            feed_dict.update({a: d for a, d in zip([dm_input], [design_matrix])})

            feed_dict.update({a: d for a, d in zip(cnt_inputs[: len(waves[i_wave])], counter)})

            results = []
            results += sess.run(work[:len(waves[i_wave])],
                                feed_dict=feed_dict)

            for result in results:
                dti_params[thread_mask == result[1][0]] = \
                    result[0].reshape(result[0].shape[0], 12)

        evals = dti_params[..., :3]

        evals = mb._roll_evals(evals, -1)

        all_zero = (evals == 0).all(axis=0)
        ev1, ev2, ev3 = evals
        fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                            (ev2 - ev3) ** 2 +
                            (ev3 - ev1) ** 2) /
                     ((evals * evals).sum(0) + all_zero))
        fas.append(fa)

    return fas


class TfCluster(object):
    def __init__(self, worker, name="worker"):
        self.worker = worker
        self.devices = ["/job:%s/task:%d" % (name, i)
                        for i in range(len(worker))]
        self.cluster = tf.train.ClusterSpec({name: worker})
        self.server = tf.train.Server(self.cluster,
                                      job_name=name,
                                      task_index=0)

    @property
    def n_worker(self):
        return len(self.worker)

    @property
    def host(self):
        return self.worker[0]

    def partition_work_waves(self, n_jobs, use_host=True):
        waves = []

        if use_host:
            this_worker = self.devices
        else:
            this_worker = self.devices[1:]

        while n_jobs > 0:
            if n_jobs >= len(this_worker):
                waves.append(this_worker)
                n_jobs -= len(this_worker)
            else:
                waves.append(this_worker[:n_jobs])
                n_jobs = 0

        return waves