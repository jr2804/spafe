import scipy
import pytest
from mock import patch
import scipy.io.wavfile
from spafe.features.spfeats import extract_feats
from spafe.frequencies.dominant_frequencies import get_dominant_frequencies
from spafe.frequencies.fundamental_frequencies import FundamentalFrequenciesExtractor

DEBUG_MODE = False


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize('debug', [False, True])
def test_dom_freqs(mock, debug):
    """
    test the computation of dominant frequencies
    """
    sig_and_fs = [scipy.io.wavfile.read("tests/test_files/test_file_8000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_16000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_32000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_44100Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_48000Hz.wav")]
    for fs, sig in sig_and_fs:
        # test dominant frequencies extraction
        dom_freqs = get_dominant_frequencies(sig=sig,
                                             fs=fs,
                                             butter_filter=False,
                                             lower_cutoff=50,
                                             upper_cutoff=3000,
                                             nfft=512,
                                             win_len=0.025,
                                             win_hop=0.01,
                                             win_type="hamming",
                                             debug=False)
        # assert is not None
        if dom_freqs is None:
            raise AssertionError


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize('debug', [False, True])
def test_fund_freqs(mock, debug):
    """
    test the computation of fundamental frequencies.
    """
    sig_and_fs = [scipy.io.wavfile.read("tests/test_files/test_file_8000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_16000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_32000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_44100Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_48000Hz.wav")]
    for fs, sig in sig_and_fs:
        #  test fundamental frequencies extraction
        fund_freqs_extractor = FundamentalFrequenciesExtractor(debug=False)
        pitches, harmonic_rates, argmins, times = fund_freqs_extractor.main(
            sig=sig, fs=fs)
        # assert is not None
        if pitches is None:
            raise AssertionError


@patch("matplotlib.pyplot.show")
def test_extract_feats(mock_show):
    """
    test the computations of spectral features.
    """
    sig_and_fs = [scipy.io.wavfile.read("tests/test_files/test_file_8000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_16000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_32000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_44100Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_48000Hz.wav")]
    for fs, sig in sig_and_fs:
        spectral_features = extract_feats(sig=sig, fs=fs)

        # general stats
        if not len(spectral_features) == 34:
            raise AssertionError

        if not spectral_features["duration"] == (len(sig) / float(fs)):
            raise AssertionError

        for _, v in spectral_features.items():
            if v is None:
                raise AssertionError
