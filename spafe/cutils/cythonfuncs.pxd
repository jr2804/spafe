import numpy as np
cimport numpy as np


FTYPE = np.float
ctypedef np.float_t FTYPE_t


cdef int F0 = 0
cdef float FSP = 200 / 3
cdef int BARK_FREQ = 1000
cdef float BARK_PT = (BARK_FREQ - F0) / FSP
cdef float LOGSTEP = np.exp(np.log(6.4) / 27.0)


"""
Converters
"""
cpdef float cyhz2mel(float hz, int htk)
cpdef float cyhz2bark(float f)
cpdef float cyfft2bark(int fft, int fs, int nfft)
cpdef np.ndarray[FTYPE_t, ndim=2] cyfft2barkmx(float min_bark,
                                               float nyqbark,
                                               int nfft, int fs,
                                               int nfilts,
                                               int bwidth)


"""
Bark fbanks
"""
cpdef float cyFm(float fb, float fc)
cpdef np.ndarray[FTYPE_t, ndim=2] cybark_helper(str scale, int nfilts, int nfft,
                                                np.ndarray[double, ndim=1] bins,
                                                np.ndarray[double, ndim=1] bark_points)


"""
Mel & Lin fbanks
"""
cpdef np.ndarray[FTYPE_t, ndim=2] cymel_and_lin_helper(str scale,
                                                       int nfilts,
                                                       int nfft,
                                                       np.ndarray[double, ndim=1] bins)


"""
Gammatone fbanks
"""
cpdef tuple cycompute_gain(np.ndarray[FTYPE_t, ndim=1] fcs, np.ndarray[FTYPE_t, ndim=1] B,
                           np.ndarray[FTYPE_t, ndim=1] wT, float T)
