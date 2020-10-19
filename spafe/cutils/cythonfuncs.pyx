import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, log, log10, sqrt, fmin, fmax

# define numpy type
FTYPE = np.float

# Slaney's ERB Filter constants
cdef float EarQ = 9.26449
cdef float minBW = 24.7


"""
Helpers
"""
cdef float cython_min(np.ndarray[FLOAT_t, ndim=2] a):
    cdef float min = np.inf
    cdef int i, j
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] < min:
                min = a[i]
    return min


cdef float cython_max(np.ndarray[FLOAT_t, ndim=2] a):
    cdef float max = -np.inf
    cdef int i, j
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] > max:
                max = a[i]
    return max


cpdef FLOAT_t cython_sum(np.ndarray[FLOAT_t, ndim=1] A):
   cdef double [:] x = A
   cdef double sum = 0
   cdef unsigned int N = A.shape[0]
   for i in xrange(N):
     sum += x[i]
   return sum

cpdef FLOAT_t cython_avg(np.ndarray[FLOAT_t, ndim=1] A):
   cdef double [:] x = A
   cdef double sum = 0
   cdef unsigned int N = A.shape[0]
   for i in xrange(N):
     sum += x[i]
   return sum/N


cpdef FLOAT_t cython_silly_avg(np.ndarray[FLOAT_t, ndim=1] A):
   cdef unsigned int N = A.shape[0]
   return cython_avg(A)*N


"""
Utils: Converters
"""
cpdef float cyhz2mel(float hz, int htk):
    """
    Convert a value in Hertz to Mels
    """
    cdef float res

    if htk == 1:
      res = 2595 * log10(1 + hz / 700.)

    else:
      if hz < BARK_PT:
        res = (hz - F0) / FSP

      else :
        res = BARK_PT + (log(hz / BARK_FREQ) / log(LOGSTEP))
    return res


cpdef float cyhz2bark(float f):
    """
    Convert Hz frequencies to Bark acording to Wang, Sekey & Gersho, 1992.
    """
    cdef float res = 6. * np.arcsinh(f / 600.)
    return res


cpdef float cyfft2bark(int fft, int fs, int nfft):
    """
    Convert Bark frequencies to Hz.
    """
    cdef float input = (fft * fs) / (nfft + 1)
    cdef float res = cyhz2bark(input)
    return res


cpdef np.ndarray[FLOAT_t, ndim=2] cyfft2barkmx(float min_bark, float nyqbark,
                                               int nfft, int fs,int nfilts,
                                               int bwidth):
    """
    Generate a matrix of weights to combine FFT bins into Bark bins.
    """
    cdef np.ndarray[FLOAT_t, ndim=2] wts = np.zeros((nfilts, nfft), dtype=FTYPE)
    cdef float step_barks = nyqbark / (nfilts - 1)
    cdef float binbarks = cyhz2bark((fs / nfft) * np.arange(0, nfft / 2 + 1))

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
Fbanks: Bark
"""
cpdef float cyFm(float fb, float fc):
    """
    Compute a Bark filter around a certain center frequency in bark.
    """
    cdef int c1 = (fc - 2.5 <= fb) and (fb <= fc - 0.5)
    cdef int c2 = (fc - 0.5 < fb) and (fb < fc + 0.5)
    cdef int c3 = (fc + 0.5 <= fb) and (fb <= fc + 1.3)
    cdef float res = 10**(2.5 * (fb - fc + 0.5)) * c1 + 1.0 * c2 + 10**(-2.5 * (fb - fc - 0.5)) * c3
    return res


cpdef np.ndarray[FLOAT_t, ndim=2] cybark_helper(str scale, int nfilts,
                                                int nfft,
                                                np.ndarray[double, ndim=1] bins,
                                                np.ndarray[double, ndim=1] bark_points):
    """
    Compute fbank helper.
    """
    cdef np.ndarray[FLOAT_t, ndim=2] fbank = np.zeros((nfilts, nfft // 2 + 1), dtype=FTYPE)

    # init scaler
    cdef float c = 1.0 * int(scale == "descendant" or scale == "constant")

    cdef int j
    cdef float fc
    cdef float fb
    cdef float ffbfc
    cdef int i

    for j in range(2, nfilts + 2):
        # compute scaler
        if scale == "descendant":
            c = c - 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)

        elif scale == "ascendant":
            c = c + 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        for i in range(int(bins[j - 2]), int(bins[j + 2])):
            fc = bark_points[j]
            fb = cyfft2bark(i, nfilts, nfft)
            ffbfc = cyFm(fb, fc)
            fbank[j - 2, i] = c * ffbfc

    return fbank


"""
Fbanks: Mel & Lin
"""
cpdef np.ndarray[FLOAT_t, ndim=2] cymel_and_lin_helper(str scale,
                                                       int nfilts,
                                                       int nfft,
                                                       np.ndarray[double, ndim=1] bins):
    """
    Compute fbank helper.
    """
    cdef np.ndarray[FLOAT_t, ndim=2] fbank = np.zeros((nfilts, nfft // 2 + 1), dtype=FTYPE)

    # init scaler
    cdef float c = 1.0 * int(scale == "descendant" or scale == "constant")

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


"""
Fbanks: Gammatone
"""
cpdef tuple cycompute_gain(np.ndarray[FLOAT_t, ndim=1] fcs, np.ndarray[FLOAT_t, ndim=1] B,
                           np.ndarray[FLOAT_t, ndim=1] wT, float T):
    """
    Compute Gaina and matrixify computation for speed purposes.
    """
    # pre-computations for simplification
    cdef np.ndarray[FLOAT_t, ndim=1] K = np.exp(B * T)
    cdef np.ndarray[FLOAT_t, ndim=1] Cos = np.cos(2 * fcs * np.pi * T)
    cdef np.ndarray[FLOAT_t, ndim=1] Sin = np.sin(2 * fcs * np.pi * T)
    cdef float Smax = sqrt(3 + 2**(3 / 2))
    cdef float Smin = sqrt(3 - 2**(3 / 2))

    # define A matrix rows
    cdef np.ndarray[FLOAT_t, ndim=1] A11 = (Cos + Smax * Sin) / K
    cdef np.ndarray[FLOAT_t, ndim=1] A12 = (Cos - Smax * Sin) / K
    cdef np.ndarray[FLOAT_t, ndim=1] A13 = (Cos + Smin * Sin) / K
    cdef np.ndarray[FLOAT_t, ndim=1] A14 = (Cos - Smin * Sin) / K

    # Compute gain (vectorized)
    cdef np.ndarray[FLOAT_t, ndim=2] A = np.array([A11, A12, A13, A14])
    Kj = np.exp(1j * wT)
    Kjmat = np.array([Kj, Kj, Kj, Kj]).T
    G = 2 * T * Kjmat * (A.T - Kjmat)
    Coe = -2 / K**2 - 2 * Kj**2 + 2 * (1 + Kj**2) / K
    Gain = np.abs(G[:, 0] * G[:, 1] * G[:, 2] * G[:, 3] * Coe**-4)
    return A, Gain


"""
Features: PNCC
"""
cpdef np.ndarray[FLOAT_t, ndim=2] cymedium_time_power_calculation(np.ndarray[FLOAT_t, ndim=2] power_stft_signal, int M=2):
    cdef np.ndarray[FLOAT_t, ndim=2] medium_time_power = np.zeros_like(power_stft_signal)
    power_stft_signal = np.pad(power_stft_signal, [(M, M), (0, 0)], 'constant')

    cdef int i = 0
    for i in range(medium_time_power.shape[0]):
        medium_time_power[i, :] = sum([1 / (2 * M + 1) * power_stft_signal[i + k - M, :] for k in range(2 * M + 1)])
    return medium_time_power

cpdef np.ndarray[FLOAT_t, ndim=2] cyweight_smoothing(np.ndarray[FLOAT_t, ndim=2] final_output,
                                                     np.ndarray[FLOAT_t, ndim=2] medium_time_power,
                                                     int N=4, int L=128):
    cdef np.ndarray[FLOAT_t, ndim=2] spectral_weight_smoothing = np.zeros_like(final_output)
    cdef np.npy_intp m = 0
    cdef np.npy_intp l = 0
    cdef int l1 = 0
    cdef int l2 = 0
    cdef float e = 0
    cdef float s = 0

    for m in range(final_output.shape[0]):
        for l in range(final_output.shape[1]):
            l_1 = (l - N) * ((l - N) >= 1) + 1 * ((l - N) < 1)
            l_2 = (l + N) * ((l + N) <= L) + L * ((l + N) > L)

            e = 1 / (l_2 - l_1 + 1)
            s = sum([(final_output[m, l_] / medium_time_power[m, l_]) for l_ in range(l_1, l_2)])
            spectral_weight_smoothing[m, l] = e * s

    return spectral_weight_smoothing


cpdef np.ndarray[FLOAT_t, ndim=2] cymean_power_normalization(np.ndarray[FLOAT_t, ndim=2] transfer_function,
                                                             float lam_myu=0.999,
                                                             int L=80, int k=1):
    cdef np.ndarray[FLOAT_t, ndim=1] myu = np.zeros(shape=(transfer_function.shape[0]))
    myu[0] = 0.0001

    cdef np.ndarray[FLOAT_t, ndim=2] normalized_power = np.zeros_like(transfer_function)
    cdef np.npy_intp m = 0

    for m in range(1, transfer_function.shape[0]):
        myu[m] = lam_myu * myu[m - 1] + (1 - lam_myu) / L * sum([transfer_function[m, s] for s in range(0, L - 1)])

    normalized_power = k * transfer_function / myu[:, None]
    return normalized_power
