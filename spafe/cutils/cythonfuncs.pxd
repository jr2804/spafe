import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, exp, log, log10


FTYPE = np.float
ctypedef np.float64_t FLOAT_t


cdef int F0 = 0
cdef float FSP = 200 / 3
cdef int BARK_FREQ = 1000
cdef float BARK_PT = (BARK_FREQ - F0) / FSP
cdef float LOGSTEP = exp(log(6.4) / 27.0)

"""
Helpers
"""
cdef float cython_min(np.ndarray[FLOAT_t, ndim=2] a)
cdef float cython_max(np.ndarray[FLOAT_t, ndim=2] a)
cpdef FLOAT_t cython_sum(np.ndarray[FLOAT_t, ndim=1] a)
cpdef FLOAT_t cython_avg(np.ndarray[FLOAT_t, ndim=1] a)

"""
Converters
"""
cpdef float cyhz2mel(float hz, int htk)
cpdef float cyhz2bark(float f)
cpdef float cyfft2bark(int fft, int fs, int nfft)
cpdef np.ndarray[FLOAT_t, ndim=2] cyfft2barkmx(float min_bark,
                                               float nyqbark,
                                               int nfft, int fs,
                                               int nfilts,
                                               int bwidth)


"""
Bark fbanks
"""
cpdef float cyFm(float fb, float fc)
cpdef np.ndarray[FLOAT_t, ndim=2] cybark_helper(str scale, int nfilts, int nfft,
                                                np.ndarray[double, ndim=1] bins,
                                                np.ndarray[double, ndim=1] bark_points)


"""
Mel & Lin fbanks
"""
cpdef np.ndarray[FLOAT_t, ndim=2] cymel_and_lin_helper(str scale,
                                                       int nfilts,
                                                       int nfft,
                                                       np.ndarray[double, ndim=1] bins)


"""
Gammatone fbanks
"""
cpdef tuple cycompute_gain(np.ndarray[FLOAT_t, ndim=1] fcs, np.ndarray[FLOAT_t, ndim=1] B,
                           np.ndarray[FLOAT_t, ndim=1] wT, float T)


"""
Featues: PNCC
"""
cpdef np.ndarray[FLOAT_t, ndim=2] cymedium_time_power_calculation(np.ndarray[FLOAT_t, ndim=2] power_stft_signal,
                                                                  int M=*)
cpdef np.ndarray[FLOAT_t, ndim=2] cyweight_smoothing(np.ndarray[FLOAT_t, ndim=2] final_output,
                                                     np.ndarray[FLOAT_t, ndim=2] medium_time_power,
                                                     int N=*, int L=*)
cpdef np.ndarray[FLOAT_t, ndim=2] cymean_power_normalization(np.ndarray[FLOAT_t, ndim=2] transfer_function,
                                                             float lam_myu=*, int L=*, int k=*)
