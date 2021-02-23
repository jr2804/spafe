import pytest
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.rplp import rplp, plp
from spafe.utils.spectral import stft, display_stft


@pytest.mark.test_id(207)
@pytest.mark.parametrize('sig_and_fs', [scipy.io.wavfile.read("tests/test_files/test_file_8000Hz.wav"),
                                        scipy.io.wavfile.read("tests/test_files/test_file_16000Hz.wav"),
                                        scipy.io.wavfile.read("tests/test_files/test_file_32000Hz.wav"),
                                        scipy.io.wavfile.read("tests/test_files/test_file_44100Hz.wav"),
                                        scipy.io.wavfile.read("tests/test_files/test_file_48000Hz.wav")])
@pytest.mark.parametrize('num_ceps', [13, 15])
@pytest.mark.parametrize('pre_emph', [False, True])
@pytest.mark.parametrize('modelorder', [0, 23])
def test_rplp(sig_and_fs, num_ceps, pre_emph, modelorder):
    """
    test RPLP features module for the following:
        - check that the returned number of cepstrums is correct.
    """
    sig, fs = sig_and_fs[1], sig_and_fs[0]

    # compute plps
    plps = plp(sig, fs, num_ceps, pre_emph)

    # assert number of returned cepstrum coefficients
    if not plps.shape[1] == num_ceps:
        raise AssertionError

    # compute bfccs
    rplps = rplp(sig, fs, num_ceps, pre_emph, modelorder)

    # assert number of returned cepstrum coefficients
    if not rplps.shape[1] == num_ceps:
        raise AssertionError
