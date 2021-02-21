import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.features.pncc import pncc
from spafe.utils.exceptions import ParameterError
from spafe.utils.cepstral import cms, cmvn, lifter_ceps


@pytest.fixture
def sig():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[1]


@pytest.fixture
def fs():
    __EXAMPLE_FILE = 'test.wav'
    return scipy.io.wavfile.read(__EXAMPLE_FILE)[0]


@pytest.mark.test_id(202)
@pytest.mark.parametrize('num_ceps', [13, 39])
@pytest.mark.parametrize('nfilts', [24, 36])
@pytest.mark.parametrize('nfft', [512, 1024])
@pytest.mark.parametrize('low_freq', [-5, 0, 300])
@pytest.mark.parametrize('high_freq', [3000])
@pytest.mark.parametrize('dct_type', [1, 2, 4])
@pytest.mark.parametrize('use_energy', [False, True])
@pytest.mark.parametrize('lifter', [0, 5])
@pytest.mark.parametrize('normalize', [False, True])
def test_pncc(sig, fs, num_ceps, nfilts, nfft, low_freq, high_freq, dct_type,
              use_energy, lifter, normalize):
    """
    test PNCC features module for the following:
        - check if ParameterErrors are raised for:
                - nfilts < num_ceps
                - negative low_freq value
                - high_freq > fs / 2
        - check that the returned number of cepstrums is correct.
        - check the use energy functionality.
        - check normalization.
        - check liftering.
    """
    # check error for number of filters is smaller than number of cepstrums
    if (low_freq < 0) or (high_freq > fs / 2) or (nfilts < num_ceps) :
        with pytest.raises(ParameterError):
            pnccs = pncc(sig=sig,
                         fs=fs,
                         num_ceps=num_ceps,
                         nfilts=nfilts,
                         nfft=nfft,
                         low_freq=low_freq,
                         high_freq=high_freq,)
    else:
        # compute features
        pnccs = pncc(sig=sig,
                     fs=fs,
                     num_ceps=num_ceps,
                     nfilts=nfilts,
                     nfft=nfft,
                     low_freq=low_freq,
                     high_freq=high_freq,
                     dct_type=dct_type,
                     use_energy=use_energy,
                     lifter=lifter,
                     normalize=normalize)

        # assert number of returned cepstrum coefficients
        if not pnccs.shape[1] == num_ceps:
            raise AssertionError

        # check use energy
        if use_energy:
            pnccs_energy = pnccs[:, 0]
            gfccs_energy = pncc(sig=sig,
                                fs=fs,
                                num_ceps=num_ceps,
                                nfilts=nfilts,
                                nfft=nfft,
                                low_freq=low_freq,
                                high_freq=high_freq,
                                dct_type=dct_type,
                                use_energy=use_energy,
                                lifter=lifter,
                                normalize=normalize)[:, 0]

            np.testing.assert_almost_equal(pnccs_energy, gfccs_energy, 3)

        # check normalize
        if normalize:
            np.testing.assert_almost_equal(
                pnccs,
                cmvn(
                    cms(
                        pncc(sig=sig,
                             fs=fs,
                             num_ceps=num_ceps,
                             nfilts=nfilts,
                             nfft=nfft,
                             low_freq=low_freq,
                             high_freq=high_freq,
                             dct_type=dct_type,
                             use_energy=use_energy,
                             lifter=lifter,
                             normalize=False))), 3)
        else:
            # check lifter
            if lifter > 0:
                np.testing.assert_almost_equal(
                    pnccs,
                    lifter_ceps(
                        pncc(sig=sig,
                             fs=fs,
                             num_ceps=num_ceps,
                             nfilts=nfilts,
                             nfft=nfft,
                             low_freq=low_freq,
                             high_freq=high_freq,
                             dct_type=dct_type,
                             use_energy=use_energy,
                             lifter=False,
                             normalize=normalize), lifter), 3)
