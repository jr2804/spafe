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
cpdef float chz2mel(float hz, int htk)
cpdef float chz2bark(float f)
cpdef float cfft2bark(int fft, int fs, int nfft)
cpdef np.ndarray[FTYPE_t, ndim=2] cfft2barkmx(float min_bark, float nyqbark, int nfft, int fs, int nfilts, int bwidth)


"""
Bark fbanks
"""
cpdef float cFm(float fb, float fc)
cpdef np.ndarray[FTYPE_t, ndim=2] bark_helper(str scale, int nfilts, int nfft, np.ndarray[double, ndim=1] bins, np.ndarray[double, ndim=1] bark_points)


"""
Mel & Lin fbanks
"""
cpdef np.ndarray[FTYPE_t, ndim=2] mel_and_lin_helper(str scale, int nfilts, int nfft, np.ndarray[double, ndim=1] bins)
