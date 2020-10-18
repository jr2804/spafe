#############################################################################################
#                           Mel-filter-banks implementation
#############################################################################################
import numpy as np
from ..utils.converters import hz2mel, mel2hz
from ..utils.exceptions import ParameterError, ErrorMsgs
from ..cutils.cythonfuncs import cyhz2mel, cymel_and_lin_helper


def mel_filter_banks(nfilts=20,
                     nfft=512,
                     fs=16000,
                     low_freq=0,
                     high_freq=None,
                     scale="constant"):
    """
    Compute Mel-filterbanks.The filters are stored in the rows, the columns
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
        scale    (str)  : choose if mx bins amplitudes sum up to one or are constants.
                          Default is "constant"

    Returns:
        a numpy array of size nfilts * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    # compute points evenly spaced in mels (ponts are in Hz)
    low_mel = cyhz2mel(low_freq, 1)
    high_mel = cyhz2mel(high_freq, 1)
    mel_points = np.linspace(low_mel, high_mel, nfilts + 2)

    # we use fft bins, so we have to convert from Hz to fft bin number
    bins = np.floor((nfft + 1) * mel2hz(mel_points) / fs)

    # compute amps of fbanks
    fbank = cymel_and_lin_helper(scale, nfilts, nfft, bins)
    return np.abs(fbank)


def inverse_mel_filter_banks(nfilts=20,
                             nfft=512,
                             fs=16000,
                             low_freq=0,
                             high_freq=None,
                             scale="constant"):
    """
    Compute inverse Mel-filterbanks. The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilt     (int) : the number of filters in the filterbank.
                          (Default 20)
        nfft      (int) : the FFT size.
                          (Default is 512)
        fs        (int) : sample rate/ sampling frequency of the signal.
                          (Default 16000 Hz)
        low_freq  (int) : lowest band edge of mel filters.
                          (Default 0 Hz)
        high_freq (int) : highest band edge of mel filters.
                          (Default samplerate/2)
        scale    (str)  : choose if mx bins amplitudes sum up to one or are constants.
                          Default is "const"

    Returns:
        a numpy array of size nfilt * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])

    # inverse scaler value
    scales = {
        "ascendant": "descendant",
        "descendant": "ascendant",
        "constant": "constant"
    }
    iscale = scales[scale]
    # generate inverse mel fbanks by inversing regular mel fbanks
    imel_fbanks = mel_filter_banks(nfilts=nfilts,
                                   nfft=nfft,
                                   fs=fs,
                                   low_freq=low_freq,
                                   high_freq=high_freq,
                                   scale=iscale)
    # inverse regular filter banks
    for i, pts in enumerate(imel_fbanks):
        imel_fbanks[i] = pts[::-1]

    return np.abs(imel_fbanks)
