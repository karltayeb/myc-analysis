# cython: profile=True
import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel
from libc.math cimport log, M_PI

@cython.boundscheck(False)
cpdef void dot(np.float64_t[:, :] A, np.float64_t[:, :] B, np.float64_t[:, :] result) nogil:
    cdef int n = A.shape[0]
    cdef int m = B.shape[0]
    cdef int p = B.shape[1]
    cdef int num_threads
    cdef int i, j, k

    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i, k] = result[i, k] + (A[i, j] * B[j, k])


# cython: profile=True

@cython.boundscheck(False)
cpdef double inner(np.float64_t[:] A, np.float64_t[:] B) nogil:
    cdef int n = A.shape[0]
    cdef double inner = 0
    cdef int i

    for i in range(n):
        inner = inner + (A[i] * B[i])

    return inner

@cython.boundscheck(False)
cpdef void outer(np.float64_t[:] A, np.float64_t[:] B, np.float64_t[:, :] result) nogil:
    cdef int n = A.shape[0]
    cdef int i, j

    for i in range(n):
        for j in range(n):
            result[i, j] = A[i] * B[j]


@cython.boundscheck(False)
cpdef double trace(np.float64_t[:, :] matrix) nogil:
    cdef int dim = matrix.shape[0]
    cdef int i
    cdef double trace = 0
    for i in range(dim):
        trace = trace + matrix[i, i]
    return trace

@cython.boundscheck(False)
cpdef double elementwise_sum(double[:, :, :] matrix, double[:, :] result):
    """
    takes sum of 3d matrix along first axis
    resturns 2d matrix
    """
    cdef int ax1_dim = matrix.shape[0]
    cdef int ax2_dim = matrix.shape[1]
    cdef int ax3_dim = matrix.shape[2]

    cdef int i, j, k
    for i in range(ax2_dim):
        for j in range(ax3_dim):
            result[i, j] = 0
            for k in range(ax1_dim):
                result[i, j] = result[i, j] + matrix[k, i, j]

@cython.boundscheck(False)
cpdef sum(np.float64_t[:, :] matrix, np.float64_t[:] result, int axis):
    """
    takes sum of 3d matrix along first axis
    resturns 2d matrix
    """
    if axis == 1:
        matrix = matrix.T

    cdef int dim0 = matrix.shape[0]
    cdef int dim1 = matrix.shape[1]

    cdef int i, j
    for j in range(dim1):
        result[j] = 0
        for i in range(dim0):
            result[j] += matrix[i, j]

@cython.boundscheck(False)
cpdef double normal_logpdf(
    np.float64_t[:] x,
    np.float64_t[:] mean,
    np.float64_t[:, :] covariance,
    np.float64_t[:, :] precision):

    cdef int i, dim = x.shape[0]
    cdef np.float64_t[:] residual = np.zeros(dim, dtype=np.float64)
    cdef double quad
    cdef double const
    cdef np.float64_t[:] temp = np.zeros(dim, dtype=np.float64)

    for i in range(dim):
        residual[i] = x[i] - mean[i]

    dot(residual[None, :], precision, temp[None, :])
    quad = inner(temp, residual)

    const = log(np.linalg.det(2 * M_PI * np.asarray(covariance)))
    
    return -0.5 * (quad + const)


@cython.boundscheck(False)
cpdef double expected_normal_logpdf(
    np.float64_t[:] x,
    np.float64_t[:] mean,
    np.float64_t[:, :] covariance,
    np.float64_t[:, :] precision,
    np.float64_t[:, :] V):

    cdef int i, dim = x.shape[0]
    cdef np.float64_t[:] residual = np.zeros(dim, dtype=np.float64)
    cdef double expected_quad
    cdef double const
    cdef double result

    for i in range(dim):
        residual[i] = x[i] - mean[i]

    expected_quad = quadratic_expectation(residual, precision, V)
    const = dim * log(2 * M_PI) + log(np.linalg.det(np.asarray(covariance)))
    
    result = -0.5 * (expected_quad + const)
    return result


@cython.boundscheck(False)
cpdef double quadratic_expectation(
    np.float64_t[:] x,
    np.float64_t[:, :] A,
    np.float64_t[:, :] V
    ):
    """
    expectation of quadratic from x'Ax
    E[x'Ax] = E[x'] A E[x] + tr(AV)
    """
    cdef int dim = x.shape[0]
    cdef np.float64_t[:] temp = np.zeros(dim, dtype=np.float64)
    cdef double quad
    cdef double traceAV
    cdef np.float64_t[:,:] AV = np.zeros((dim, dim), dtype=np.float64)
    cdef double result

    dot(x[None, :], A, temp[None, :])
    quad = inner(temp, x)
    
    dot(A, V, AV)
    traceAV = trace(AV)

    result = quad + traceAV
    return result

@cython.boundscheck(False)
cpdef double quadratic(
    np.float64_t[:] x,
    np.float64_t[:, :] A,
    np.float64_t[:, :] V
    ):
    """
    expectation of quadratic from x'Ax
    E[x'Ax] = E[x'] A E[x] + tr(AV)
    """
    cdef int dim = x.shape[0]
    cdef np.float64_t[:] temp = np.zeros(dim, dtype=np.float64)
    cdef double quad

    dot(x[None, :], A, temp[None, :])
    quad = inner(temp, x)

    return quad


@cython.boundscheck(False) # turn off bounds-checking for entire function
cpdef np.float64_t[:, :] woodbury_inversion(
    np.float64_t[:, :] Ainv,
    np.float64_t[:, :] U,
    np.float64_t[:, :] Cinv,
    np.float64_t[:, :] V):
    
    cdef int ndim = Ainv.shape[0]
    cdef int mdim = Cinv.shape[0]
    cdef np.float64_t[:, :] inverse
    cdef np.float64_t[:, :] sub

    sub = Cinv + np.linalg.multi_dot([
        V,
        Ainv,
        U
    ])

    inverse = Ainv - np.linalg.multi_dot([
        Ainv,
        U,
        np.linalg.pinv(sub),
        V,
        Ainv
    ])

    return inverse


@cython.boundscheck(False) # turn off bounds-checking for entire function
cpdef np.float64_t[:, :] _expected_log_likelihoods(
    np.float64_t[:, :, :] data,
    np.float64_t[:, :] observation_precision,
    np.float64_t[:, :] projected_state_means,
    np.float64_t[:, :, :] projected_state_covariances,
    np.float64_t logdet):
    

    cdef int n_obs = data.shape[0]
    cdef int t_timepoints = data.shape[1]
    cdef int n_dim_obs = data.shape[2]

    cdef np.float64_t[:, :] expected_log_likelihoods = np.zeros((n_obs, t_timepoints))

    cdef np.float64_t[:] residual = np.zeros(n_dim_obs)

    cdef double traceAV
    cdef np.float64_t[:,:] AV = np.zeros((n_dim_obs, n_dim_obs), dtype=np.float64)
    
    cdef int n, t, i
    
    for t in range(t_timepoints):
        dot(observation_precision, projected_state_covariances[t], AV)
        traceAV = trace(AV)
        for n in range(n_obs):
            for i in range(n_dim_obs):
                residual[i] = data[n, t, i] - projected_state_means[t, i]
            expected_log_likelihoods[n, t] =  expected_log_likelihoods[n, t] + \
                quadratic(
                    x=residual,
                    A=observation_precision,
                    V=projected_state_covariances[t]
                ) + traceAV + logdet

            expected_log_likelihoods[n, t] = 0.5 * expected_log_likelihoods[n, t]

    return expected_log_likelihoods