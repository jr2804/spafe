import scipy
import spafe
import pytest
import numpy as np
from mock import patch
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.lfcc import lfcc
from spafe.fbanks import linear_fbanks
from spafe.utils.exceptions import assert_function_availability


def test_functions_availability():
    # Cheching the availibility of functions in the chosen attribute
    assert_function_availability(hasattr(spafe.utils.vis, 'visualize_fbanks'))
    assert_function_availability(hasattr(spafe.utils.vis, 'visualize_features'))
    assert_function_availability(hasattr(spafe.utils.vis, 'plot'))
    assert_function_availability(hasattr(spafe.utils.vis, 'spectogram'))


@patch("matplotlib.pyplot.show")
def test_visualize_fbanks(mock_show):
    # compute filterbanks
    lin_filbanks = linear_fbanks.linear_filter_banks()
    vis.visualize_fbanks(fbanks=lin_filbanks,
                         ylabel="Amplitude",
                         xlabel="Frequency (Hz)")


@patch("matplotlib.pyplot.show")
def test_visualize_features(mock):
    sig_and_fs = [scipy.io.wavfile.read("tests/test_files/test_file_8000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_16000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_32000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_44100Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_48000Hz.wav")]
    for fs, sig in sig_and_fs:
        lfccs = lfcc(sig=sig, fs=fs)
        vis.visualize_features(feats=lfccs,
                               ylabel='LFCC Index',
                               xlabel='Frame Index',
                               cmap='viridis')


@patch("matplotlib.pyplot.show")
def test_plot(mock_show):
    y = np.arange(10)
    vis.plot(y=y, ylabel="y", xlabel="x")


@patch("matplotlib.pyplot.show")
def test_spectogram(mock):
    sig_and_fs = [scipy.io.wavfile.read("tests/test_files/test_file_8000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_16000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_32000Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_44100Hz.wav"),
                  scipy.io.wavfile.read("tests/test_files/test_file_48000Hz.wav")]
    for fs, sig in sig_and_fs:
        vis.spectogram(sig, fs)
