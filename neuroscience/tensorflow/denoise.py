import numpy as np
import tensorflow as tf


def estimate_sigma(arr, depth):
    k = np.ones((3, 3, depth, 1), dtype=np.float32) / 2
    k[0, 0, :, :] = 0.
    k[2, 0, :, :] = 0.
    k[0, 2, :, :] = 0.
    k[2, 2, :, :] = 0.
    k[1, 1, :, :] = 0.

    k = tf.constant(k)
    arr = tf.to_float(arr)

    out = tf.nn.depthwise_conv2d(arr, k, [1, 1, 1, 1], 'SAME')

    out += tf.transpose(tf.nn.depthwise_conv2d(tf.transpose(arr,
                                                            perm=[2, 0, 1, 3]),
                                               k, [1, 1, 1, 1], 'SAME'),
                        perm=[1, 2, 0, 3])

    out += tf.transpose(tf.nn.depthwise_conv2d(tf.transpose(arr,
                                                            perm=[1, 2, 0, 3]),
                                               k, [1, 1, 1, 1], 'SAME'),
                        perm=[2, 0, 1, 3])

    out /= 6.

    f = tf.constant(np.sqrt(6./7.).astype(np.float32))
    out = f * (arr - out)
    out = tf.sqrt(tf.reduce_mean(tf.square(out), [0, 1, 2]))

    return out


def nlmeans(arr, sigmas, sb, depth, p=1, b=5):

    sigma = tf.ones_like(arr) * sigmas

    arr = tf.pad(arr, [[b, b], [b, b], [b, b], [0, 0]], "REFLECT")

    # arr = tf.to_float(arr)
    sumw = tf.zeros(sb)
    new_values = tf.zeros(sb)
    patch_vol_size = (2*p+1)**3

    add_filter2d = tf.ones([2*p+1, 2*p+1, depth, 1])
    add_filter1d = tf.ones([2*p+1, 1, depth, 1])

    center_block = arr[b - p: b - p + sb[0] + 2 * p,
                       b - p: b - p + sb[1] + 2 * p,
                       b - p: b - p + sb[2] + 2 * p, :]

    sigma_c = tf.nn.depthwise_conv2d(tf.pad(sigma, [[b, b], [b, b], [b, b], [0, 0]], "REFLECT"),
                                     add_filter2d, [1, 1, 1, 1], 'SAME')
    sigma_c = tf.nn.depthwise_conv2d(tf.transpose(sigma_c, perm=[2, 0, 1, 3]),
                                     add_filter1d, [1, 1, 1, 1], 'SAME')
    sigma_c = tf.transpose(sigma_c, perm=[1, 2, 0, 3])
    denom = tf.sqrt(2.) * tf.square(sigma_c) / patch_vol_size

    for m in range(p, 2 * b + 1 - p):
        for n in range(p, 2 * b + 1 - p):
            for o in range(p, 2 * b + 1 - p):

                this_block = arr[m - p: m - p + sb[0] + 2 * p,
                                 n - p: n - p + sb[1] + 2 * p,
                                 o - p: o - p + sb[2] + 2 * p, :]

                d = tf.square(center_block - this_block)

                summs = tf.nn.depthwise_conv2d(d, add_filter2d, [1, 1, 1, 1],
                                               'SAME')
                summs = tf.nn.depthwise_conv2d(
                    tf.transpose(summs, perm=[2, 0, 1, 3]), add_filter1d,
                    [1, 1, 1, 1], 'SAME')

                summs = tf.transpose(summs, perm=[1, 2, 0, 3])[p: p+sb[0],
                                                               p: p+sb[1],
                                                               p: p+sb[2],
                                                               :]

                ws = tf.exp(-summs / denom[m: m + sb[0],
                                           n: n + sb[1],
                                           o: o + sb[2],
                                           :])

                sumw += ws
                new_values += ws * tf.square(arr[m: m + sb[0],
                                                 n: n + sb[1],
                                                 o: o + sb[2],
                                                 :])

    new_values *= tf.to_float(tf.greater(sumw, 0))
    new_values /= (sumw + tf.to_float(tf.equal(sumw, 0)))
    new_values -= 2 * sigma
    new_values *= tf.to_float(tf.greater(new_values, 0))
    return tf.sqrt(new_values)


