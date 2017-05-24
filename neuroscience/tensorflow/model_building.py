import dipy.core.gradients as dpg
import dipy.reconst.dti as dti
import nibabel as nib
import numpy as np
import tensorflow as tf

MIN_POSITIVE_SIGNAL = 0.0001
tol = 1e-6

_lt_indices = np.array([[0, 1, 3],
                        [1, 2, 4],
                        [3, 4, 5]])


def pinv(a, rcond=1e-15):
    """
    IMPORTED FROM DIPY - http://nipy.org/dipy/
        
    """
    swap = np.arange(a.ndim)
    swap[[-2, -1]] = swap[[-1, -2]]
    u, s, v = np.linalg.svd(a, full_matrices=False)
    cutoff = np.maximum.reduce(s, axis=-1, keepdims=True) * rcond
    mask = s > cutoff
    s[mask] = 1. / s[mask]
    s[~mask] = 0
    return np.einsum('...ij,...jk',
                     np.transpose(v, swap) * s[..., None, :],
                     np.transpose(u, swap))


def _roll_evals(evals, axis=-1):
    """
    IMPORTED FROM DIPY - http://nipy.org/dipy/
        
    Helper function to check that the evals provided to functions calculating
    tensor statistics have the right shape
    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor. shape should be (...,3).
    axis : int
        The axis of the array which contains the 3 eigenvals. Default: -1
    Returns
    -------
    evals : array-like
        Eigenvalues of a diffusion tensor, rolled so that the 3 eigenvals are
        the last axis.
    """
    if evals.shape[-1] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[-1])
        raise ValueError(msg)

    evals = np.rollaxis(evals, axis)

    return evals


def eig_from_lo_tri(data, min_diffusivity=0):
    """
    IMPORTED FROM DIPY - http://nipy.org/dipy/
    
    Calculates tensor eigenvalues/eigenvectors from an array containing the
    lower diagonal form of the six unique tensor elements.
    Parameters
    ----------
    data : array_like (..., 6)
        diffusion tensors elements stored in lower triangular order
    min_diffusivity : float
        See decompose_tensor()
    Returns
    -------
    dti_params : array (..., 12)
        Eigen-values and eigen-vectors of the same array.
    """
    data = np.asarray(data)
    evals, evecs = decompose_tensor(data[..., _lt_indices],
                                    min_diffusivity=min_diffusivity)
    dti_params = np.concatenate((evals[..., None, :], evecs), axis=-2)
    return dti_params.reshape(data.shape[:-1] + (12, ))


def decompose_tensor(tensor, min_diffusivity=0):
    """ 
    IMPORTED FROM DIPY - http://nipy.org/dipy/
        
    Returns eigenvalues and eigenvectors given a diffusion tensor
    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors (Basser et al., 1994a).
    Parameters
    ----------
    tensor : array (..., 3, 3)
        Hermitian matrix representing a diffusion tensor.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.
    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[..., :, j] is associated with
        eigvals[..., j])
    """
    # outputs multiplicity as well so need to unique
    eigenvals, eigenvecs = np.linalg.eigh(np.asarray(tensor), 'L')

    # need to sort the eigenvalues and associated eigenvectors
    if eigenvals.ndim == 1:
        # this is a lot faster when dealing with a single voxel
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
    else:
        # temporarily flatten eigenvals and eigenvecs to make sorting easier
        shape = eigenvals.shape[:-1]
        eigenvals = eigenvals.reshape(-1, 3)
        eigenvecs = eigenvecs.reshape(-1, 3, 3)
        size = eigenvals.shape[0]
        order = eigenvals.argsort()[:, ::-1]
        xi, yi = np.ogrid[:size, :3, :3][:2]
        eigenvecs = eigenvecs[xi, yi, order[:, None, :]]
        xi = np.ogrid[:size, :3][0]
        eigenvals = eigenvals[xi, order]
        eigenvecs = eigenvecs.reshape(shape + (3, 3))
        eigenvals = eigenvals.reshape(shape + (3, ))
    eigenvals = eigenvals.clip(min=min_diffusivity)

    return eigenvals, eigenvecs


# TESTING
def model_building_stand_alone(path):
    img = nib.load(path + "/100307_data_tiny.nii.gz")
    data = img.get_data()
    img_mask = nib.load(path + "/100307_mask.nii.gz")
    mask = img_mask.get_data().astype(np.bool)

    gtab = dpg.gradient_table(path + '/100307_bvals_tiny',
                              path + '/100307_bvecs_tiny',
                              b0_threshold=10)

    data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))
    data_in_mask = np.maximum(data_in_mask, MIN_POSITIVE_SIGNAL)

    ten_model = dti.TensorModel(gtab)

    design_matrix = ten_model.design_matrix
    U, S, V = np.linalg.svd(design_matrix, False)
    ols_fit = np.dot(U, U.T)
    log_s = np.log(data_in_mask)
    w = np.exp(np.einsum('...ij,...j', ols_fit, log_s))

    fit_result = np.einsum('...ij,...j',
                  pinv(design_matrix * w[..., None]),
                  w * log_s)
    params_in_mask = eig_from_lo_tri(fit_result,
                                     min_diffusivity=tol / -design_matrix.min())

    dti_params = np.zeros(data.shape[:-1] + (12,))
    dti_params[mask, :] = params_in_mask

    evals = dti_params[..., :3]

    evals = _roll_evals(evals, -1)

    all_zero = (evals == 0).all(axis=0)
    ev1, ev2, ev3 = evals
    fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                        (ev2 - ev3) ** 2 +
                        (ev3 - ev1) ** 2) /
                 ((evals * evals).sum(0) + all_zero))

    return fa


# TESTING
def model_building_stand_alone_tf(path):
    img = nib.load(path + "/100307_data_tiny.nii.gz")
    data = img.get_data()
    img_mask = nib.load(path + "/100307_mask.nii.gz")
    mask = img_mask.get_data().astype(np.bool)

    gtab = dpg.gradient_table(path + '/100307_bvals_tiny',
                              path + '/100307_bvecs_tiny',
                              b0_threshold=10)
    data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))
    data_in_mask = np.maximum(data_in_mask, MIN_POSITIVE_SIGNAL)

    ten_model = dti.TensorModel(gtab)

    design_matrix = ten_model.design_matrix

    print(data_in_mask.shape, design_matrix.shape)

    sess = tf.InteractiveSession()
    params_in_mask = model_building(data_in_mask, design_matrix).eval()

    params_in_mask = params_in_mask.reshape(params_in_mask.shape[0], 12)
    dti_params = np.zeros(data.shape[:-1] + (12,))
    dti_params[mask, :] = params_in_mask

    evals = dti_params[..., :3]

    evals = _roll_evals(evals, -1)

    all_zero = (evals == 0).all(axis=0)
    ev1, ev2, ev3 = evals
    fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                        (ev2 - ev3) ** 2 +
                        (ev3 - ev1) ** 2) /
                 ((evals * evals).sum(0) + all_zero))

    return fa


def model_building(data_in_mask, design_matrix, index=None):
    S, U, V = tf.svd(design_matrix, compute_uv=True, full_matrices=False)
    ols_fit = tf.matmul(U, tf.transpose(U))
    log_s = tf.log(tf.to_double(data_in_mask))
    w = tf.exp(tf.transpose(tf.matmul(ols_fit, tf.transpose(log_s))))

    # pinv
    a = design_matrix * w[..., None]
    S, U, V = tf.svd(a, full_matrices=False, compute_uv=True)

    t_cutoff = tf.reduce_max(S, reduction_indices=[-1], keep_dims=True) * 1e-15
    mask = S > t_cutoff

    S = tf.to_double(mask) / S
    U = tf.transpose(U, [0, 2, 1])
    V = tf.transpose(V, [0, 2, 1])

    temp1 = V * S[..., None, :]

    pinv = tf.batch_matmul(temp1, U)

    fit_result = tf.transpose(tf.reduce_sum(
        tf.transpose(pinv, [1, 0, 2]) * (w * log_s), reduction_indices=[2]))

    # eig_from_lo_tri
    fit_result = tf.concat(1, [tf.transpose(tf.gather(tf.transpose(fit_result), _lt_indices[0])),
                               tf.transpose(tf.gather(tf.transpose(fit_result), _lt_indices[1])),
                               tf.transpose(tf.gather(tf.transpose(fit_result), _lt_indices[2]))])
    fit_result = tf.reshape(fit_result, (-1, 3, 3))

    evals, evecs = tf.self_adjoint_eig(fit_result)
    evals = tf.clip_by_value(evals, clip_value_min=tol / -tf.reduce_min(design_matrix),
                             clip_value_max=np.inf)
    params_in_mask = tf.concat(-2, (evals[..., None, :], evecs))

    if index is not None:
        return params_in_mask, index
    else:
        return params_in_mask
