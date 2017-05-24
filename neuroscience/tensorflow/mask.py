from skimage.filters import threshold_otsu as otsu
from dipy.segment.mask import multi_median
import numpy as np
import tensorflow as tf


def image_filtering_mean(arr, b0s):
    # dm = tf.gather(tf.transpose(arr, perm=[3, 0, 1, 2]), tf.where(b0s))

    indices = tf.to_int32(tf.where(b0s))
    # Use tf.shape() to make this work with dynamic shapes.
    n_indices = tf.shape(indices)[0]

    # Offset to add to each row in indices. We use `tf.expand_dims()` to make
    # this broadcast appropriately.

    # Convert indices and logits into appropriate form for `tf.gather()`.
    offset = tf.expand_dims(tf.range(0, tf.reduce_prod(tf.shape(arr)),
                                     tf.shape(arr)[-1]), 1)
    flattened_indices = tf.reshape(tf.reshape(indices, [-1])+offset, [-1])
    flattened_arr = tf.reshape(arr, [-1])

    selected_rows = tf.gather(flattened_arr, flattened_indices)

    filtered_arr = tf.reshape(selected_rows,
                              tf.concat(0, [tf.shape(arr)[:-1], [n_indices]]))

    mean_b0 = tf.reduce_mean(filtered_arr,
                             reduction_indices=-1,
                             name="mean_b0")

    return mean_b0


def image_filtering(arr, b0s, index=None):
    # dm = tf.gather(tf.transpose(arr, perm=[3, 0, 1, 2]), tf.where(b0s))

    indices = tf.to_int32(tf.where(b0s))
    # Use tf.shape() to make this work with dynamic shapes.
    n_indices = tf.shape(indices)[0]

    # Offset to add to each row in indices. We use `tf.expand_dims()` to make
    # this broadcast appropriately.

    # Convert indices and logits into appropriate form for `tf.gather()`.
    offset = tf.expand_dims(tf.range(0, tf.reduce_prod(tf.shape(arr)),
                                     tf.shape(arr)[-1]), 1)
    flattened_indices = tf.reshape(tf.reshape(indices, [-1])+offset, [-1])
    flattened_arr = tf.reshape(arr, [-1])

    selected_rows = tf.gather(flattened_arr, flattened_indices)

    filtered_arr = tf.reshape(selected_rows,
                              tf.concat(0, [tf.shape(arr)[:-1], [n_indices]]))

    if index is not None:
        return filtered_arr, index
    else:
        return filtered_arr


def mean_c(arr, reduction_indices=-1):
    mean_b0 = tf.reduce_mean(arr,
                             reduction_indices=reduction_indices,
                             name="mean_b0")
    return mean_b0


def dilation(arr):
    arr_t = tf.expand_dims(arr, -1)

    add_filter2d = np.ones((3, 3, 1, 1), dtype=np.float32)
    add_filter2d[0, 0] = 0.
    add_filter2d[2, 0] = 0.
    add_filter2d[0, 2] = 0.
    add_filter2d[2, 2] = 0.
    add_filter2d[1, 1] = 10.

    add_filter2d_t = tf.constant(add_filter2d)

    add_filter1d = np.ones((3, 1, 1, 1), dtype=np.float32)
    add_filter1d[1] = 10.

    add_filter1d_t = tf.constant(add_filter1d)

    out = tf.nn.conv2d(tf.to_float(arr_t), add_filter2d_t, [1, 1, 1, 1], 'SAME')
    out = tf.nn.conv2d(tf.transpose(out, perm=(2, 0, 1, 3)), add_filter1d_t,
                       [1, 1, 1, 1], 'SAME')
    out = tf.transpose(out, perm=(1, 2, 0, 3)) >= 10.
    return out


def median_otsu(mean_b0, median_radius=4, numpass=2, dilate=1):

    mask = multi_median(mean_b0.eval(), median_radius, numpass)

    thresh = tf.constant(otsu(mask))
    mask = tf.constant(mask) > tf.to_float(thresh)

    for _ in range(dilate):
        mask = dilation(mask)

    return tf.squeeze(mask)
