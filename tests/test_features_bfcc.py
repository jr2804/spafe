import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.fbanks import bark_fbanks
from spafe.features.bfcc import bfcc
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
@pytest.mark.parametrize('pre_emph', [False, True])
@pytest.mark.parametrize('nfilts', [24, 36])
@pytest.mark.parametrize('nfft', [512, 1024])
@pytest.mark.parametrize('low_freq', [-5, 0, 300])
@pytest.mark.parametrize('high_freq', [1000, 5000])
@pytest.mark.parametrize('scale', ["ascendant", "descendant", "constant"])
@pytest.mark.parametrize('dct_type', [1, 2, 4])
@pytest.mark.parametrize('use_energy', [False, True])
@pytest.mark.parametrize('lifter', [0, 5])
@pytest.mark.parametrize('normalize', [False, True])
def test_bfcc(sig, fs, num_ceps, pre_emph, nfilts, nfft, low_freq, high_freq,
              scale, dct_type, use_energy, lifter, normalize):
    """
    test BFCC features module for the following:
        - check if ParameterErrors are raised for:
                - nfilts < num_ceps
                - negative low_freq value
                - high_freq > fs / 2
        - check that the returned number of cepstrums is correct.
        - check the use energy functionality.
        - check normalization.
        - check liftering.
    """
    # ParameterError checks
    if (low_freq < 0) or (high_freq > fs / 2):
        with pytest.raises(ParameterError):
            bark_filbanks = bark_fbanks.bark_filter_banks(nfilts=nfilts,
                                                          nfft=nfft,
                                                          fs=fs,
                                                          low_freq=low_freq,
                                                          high_freq=high_freq)

    if (low_freq < 0) or (high_freq > fs / 2) or (nfilts < num_ceps) :
        with pytest.raises(ParameterError):
            bfccs = bfcc(sig=sig,
                         fs=fs,
                         num_ceps=num_ceps,
                         nfilts=nfilts,
                         nfft=nfft,
                         low_freq=low_freq,
                         high_freq=high_freq)
    else:
        # check with predefined fbanks
        bark_filbanks = bark_fbanks.bark_filter_banks(nfilts=nfilts,
                                                      nfft=nfft,
                                                      fs=fs,
                                                      low_freq=low_freq,
                                                      high_freq=high_freq,
                                                      scale=scale)

        # assert that the filterbank shape is correct given nfilts and nfft
        if not bark_filbanks.shape == (nfilts, nfft // 2 + 1):
            raise AssertionError

        bfccs = bfcc(sig=sig,
                     fs=fs,
                     num_ceps=num_ceps,
                     pre_emph=pre_emph,
                     nfilts=nfilts,
                     nfft=nfft,
                     low_freq=low_freq,
                     high_freq=high_freq,
                     scale=scale,
                     dct_type=dct_type,
                     use_energy=use_energy,
                     lifter=lifter,
                     normalize=normalize,
                     fbanks=bark_filbanks)

        # assert number of returned cepstrum coefficients
        if not bfccs.shape[1] == num_ceps:
            raise AssertionError


        # check without predefined fbanks
        bfccs = bfcc(sig=sig,
                     fs=fs,
                     num_ceps=num_ceps,
                     pre_emph=pre_emph,
                     nfilts=nfilts,
                     nfft=nfft,
                     low_freq=low_freq,
                     high_freq=high_freq,
                     scale=scale,
                     dct_type=dct_type,
                     use_energy=use_energy,
                     lifter=lifter,
                     normalize=normalize)

        # assert number of returned cepstrum coefficients
        if not bfccs.shape[1] == num_ceps:
            raise AssertionError

        # check use energy
        if use_energy:
            bfccs_energy = bfccs[:, 0]
            xfccs_energy = bfcc(sig=sig,
                                fs=fs,
                                num_ceps=num_ceps,
                                pre_emph=pre_emph,
                                nfilts=nfilts,
                                nfft=nfft,
                                low_freq=low_freq,
                                high_freq=high_freq,
                                scale=scale,
                                dct_type=dct_type,
                                use_energy=use_energy,
                                lifter=lifter,
                                normalize=normalize)[:, 0]

            np.testing.assert_almost_equal(bfccs_energy, xfccs_energy, 3)

        # check normalize
        if normalize:
            np.testing.assert_almost_equal(
                bfccs,
                cmvn(
                    cms(
                        bfcc(sig=sig,
                             fs=fs,
                             num_ceps=num_ceps,
                             pre_emph=pre_emph,
                             nfilts=nfilts,
                             nfft=nfft,
                             low_freq=low_freq,
                             high_freq=high_freq,
                             scale=scale,
                             dct_type=dct_type,
                             use_energy=use_energy,
                             lifter=lifter,
                             normalize=False))), 3)
        else:
            # check lifter
            if lifter > 0:
                np.testing.assert_almost_equal(
                    bfccs,
                    lifter_ceps(
                        bfcc(sig=sig,
                             fs=fs,
                             num_ceps=num_ceps,
                             pre_emph=pre_emph,
                             nfilts=nfilts,
                             nfft=nfft,
                             low_freq=low_freq,
                             high_freq=high_freq,
                             scale=scale,
                             dct_type=dct_type,
                             use_energy=use_energy,
                             lifter=False,
                             normalize=normalize), lifter), 3)
