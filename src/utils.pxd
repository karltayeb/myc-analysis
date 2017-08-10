import numpy as np
cimport numpy as np
import cython
from cython.parallel import prange, parallel
from libc.math cimport log, M_PI
cimport openmp

cpdef void dot(np.float64_t[:, :] A, np.float64_t[:, :] B, np.float64_t[:, :] result) nogil
cpdef double inner(np.float64_t[:] A, np.float64_t[:] B) nogil
cpdef void outer(np.float64_t[:] A, np.float64_t[:] B, np.float64_t[:, :] result) nogil
cpdef double trace(np.float64_t[:, :] matrix) nogil
cpdef double elementwise_sum(double[:, :, :] matrix, double[:, :] result)

cpdef double normal_logpdf(
    np.float64_t[:] x,
    np.float64_t[:] mean,
    np.float64_t[:, :] covariance,
    np.float64_t[:, :] precision)

cpdef double expected_normal_logpdf(
    np.float64_t[:] x,
    np.float64_t[:] mean,
    np.float64_t[:, :] covariance,
    np.float64_t[:, :] precision,
    np.float64_t[:, :] V)

cpdef double quadratic_expectation(
    np.float64_t[:] x,
    np.float64_t[:, :] A,
    np.float64_t[:, :] V
    )