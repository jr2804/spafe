import numpy as np
cimport numpy as np


FTYPE = np.float


"""
Coverters
"""
cpdef float chz2mel(float hz, int htk):
    """
    Convert a value in Hertz to Mels
    """
    cdef float res

    if htk == 1:
      res = 2595 * np.log10(1 + hz / 700.)

    else:
      if hz < BARK_PT:
        res = (hz - F0) / FSP

      else :
        res = BARK_PT + (np.log(hz / BARK_FREQ) / np.log(LOGSTEP))
    return res


cpdef float chz2bark(float f):
    """
    Convert Hz frequencies to Bark acording to Wang, Sekey & Gersho, 1992.
    """
    cdef float res
    res = 6. * np.arcsinh(f / 600.)
    return res


cpdef float cfft2bark(int fft, int fs, int nfft):
    """
    Convert Bark frequencies to Hz.
    """
    cdef float res
    cdef float input
    input = (fft * fs) / (nfft + 1)
    res = chz2bark(input)
    return res


cpdef np.ndarray[FTYPE_t, ndim=2] cfft2barkmx(float min_bark, float nyqbark, int nfft, int fs, int nfilts, int bwidth):
    """
    Generate a matrix of weights to combine FFT bins into Bark bins.
    """
    cpdef np.ndarray[FTYPE_t, ndim=2] wts
    wts = np.zeros((nfilts, nfft), dtype=FTYPE)

    cdef float step_barks = nyqbark / (nfilts - 1)
    cdef float binbarks = chz2bark((fs / nfft) * np.arange(0, nfft / 2 + 1))

    cdef int i
    cdef float f_bark_mid
    cdef float lof
    cdef float hif

    for i in range(nfilts):
        f_bark_mid = min_bark + i * step_barks
        lof = binbarks - f_bark_mid - 0.5
        hif = binbarks - f_bark_mid + 0.5
        wts[i, 0:nfft // 2 + 1] = 10**np.minimum(0, np.minimum(hif, -2.5 * lof) / bwidth)
    return wts


"""
Bark fbanks
"""
cpdef float cFm(float fb, float fc):
    """
    Compute a Bark filter around a certain center frequency in bark.
    """
    cdef float res = 0.0
    if (fc - 2.5 <= fb) and (fb <= fc - 0.5):
        res = 10**(2.5 * (fb - fc + 0.5))

    elif (fc - 0.5 < fb) and (fb < fc + 0.5):
        res = 1.0

    elif (fc + 0.5 <= fb) and (fb <= fc + 1.3):
        res = 10**(-2.5 * (fb - fc - 0.5))
    return res


cpdef np.ndarray[FTYPE_t, ndim=2] bark_helper(str scale, int nfilts, int nfft, np.ndarray[double, ndim=1] bins, np.ndarray[double, ndim=1] bark_points):
    """
    Compute fbank helper.
    """
    cdef np.ndarray[FTYPE_t, ndim=2] fbank
    fbank = np.zeros((nfilts, nfft // 2 + 1), dtype=FTYPE)


    cdef float c
    # init scaler
    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c = 0

    cdef int j
    cdef float fc
    cdef float fb
    cdef float ffbfc
    cdef int i

    for j in range(2, nfilts + 2):
        # compute scaler
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)

        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        for i in range(int(bins[j - 2]), int(bins[j + 2])):
            fc = bark_points[j]
            fb = cfft2bark(i, nfilts, nfft)
            ffbfc = cFm(fb, fc)
            fbank[j - 2, i] = c * ffbfc

    return fbank


"""
Mel & Lin fbanks
"""
cpdef np.ndarray[FTYPE_t, ndim=2] mel_and_lin_helper(str scale, int nfilts, int nfft, np.ndarray[double, ndim=1] bins):
    """
    Compute fbank helper.
    """
    cdef np.ndarray[FTYPE_t, ndim=2] fbank
    fbank = np.zeros((nfilts, nfft // 2 + 1), dtype=FTYPE)

    # init scaler
    cdef float c
    c = 1.0 * int(scale == "descendant") + 0.0 * int(scale == "constant")

    cdef int j
    cdef double b0
    cdef double b1
    cdef double b2

    # compute amps of fbanks
    for j in range(0, nfilts):
        b0 = bins[j]
        b1 = bins[j + 1]
        b2 = bins[j + 2]

        # compute scaler
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)

        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        # compute fbanks
        fbank[j, int(b0):int(b1)] = c * (np.arange(int(b0), int(b1)) - int(b0)) / (b1 - b0)
        fbank[j, int(b1):int(b2)] = c * (int(b2) - np.arange(int(b1), int(b2))) / (b2 - b1)
    return fbank
