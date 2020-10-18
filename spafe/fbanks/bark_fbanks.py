##############################################################################################
#                             Bark-filter-banks implementation
##############################################################################################
import numpy as np
from ..utils.converters import fft2bark, bark2fft
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..cutils.cythonfuncs import cyFm, cyhz2bark, cybark_helper


def bark_filter_banks(nfilts=20,
                      nfft=512,
                      fs=16000,
                      low_freq=0,
                      high_freq=16000000,
                      scale="constant"):
    """
    Compute Bark-filterbanks. The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilts    (int) : the number of filters in the filterbank.
                          (Default 20)
        nfft      (int) : the FFT size.
                          (Default is 512)
        fs        (int) : sample rate/ sampling frequency of the signal.
                          (Default 16000 Hz)
        low_freq  (int) : lowest band edge of mel filters.
                          (Default 0 Hz)
        high_freq (int) : highest band edge of mel filters.
                          (Default samplerate/2)
        scale    (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                          Default is "constant"

    Returns:
        a numpy array of size nfilts * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    # init freqs
    high_freq = high_freq or np.ceil(fs / 2)
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    # compute points evenly spaced in Bark scale (points are in Bark)
    low_bark = cyhz2bark(low_freq)
    high_bark = cyhz2bark(high_freq)
    bark_points = np.linspace(low_bark, high_bark, nfilts + 4)

    # we use fft bins, so we have to convert from Bark to fft bin number
    bins = np.floor(bark2fft(bark_points))
    fbank = cybark_helper(scale, nfilts, nfft, bins, bark_points)
    return np.abs(fbank)
