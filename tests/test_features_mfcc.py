import pytest
import numpy as np
import scipy.io.wavfile
from spafe.utils import vis
from spafe.fbanks import mel_fbanks
from spafe.features.mfcc import mfcc, imfcc
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
def test_mfcc(sig, fs, num_ceps, pre_emph, nfilts, nfft, low_freq, high_freq,
              scale, dct_type, use_energy, lifter, normalize):
    """
    test MFCC features module for the following:
        - check if ParameterErrors are raised for:
                - nfilts < num_ceps
                - negative low_freq value
                - high_freq > fs / 2
        - check that the returned number of cepstrums is correct.
        - check the use energy functionality.
        - check normalization.
        - check liftering.
    """
    # check ParameterErrors
    if (low_freq < 0) or (high_freq > fs / 2) :
        with pytest.raises(ParameterError):
            # compute the Mel filterbanks
            mel_filbanks = mel_fbanks.mel_filter_banks(nfilts=nfilts,
                                                       nfft=nfft,
                                                       fs=fs,
                                                       low_freq=low_freq,
                                                       high_freq=high_freq,
                                                       scale=scale)

    if (low_freq < 0) or (high_freq > fs / 2) or (nfilts < num_ceps) :
        with pytest.raises(ParameterError):
            mfccs = mfcc(sig=sig,
                         fs=fs,
                         num_ceps=num_ceps,
                         nfilts=nfilts,
                         nfft=nfft,
                         low_freq=low_freq,
                         high_freq=high_freq)
    else:
        mel_filbanks = mel_fbanks.mel_filter_banks(nfilts=nfilts,
                                                   nfft=nfft,
                                                   fs=fs,
                                                   low_freq=low_freq,
                                                   high_freq=high_freq,
                                                   scale=scale)

        # assert that the filterbank shape is correct given nfilts and nfft
        if not mel_filbanks.shape == (nfilts, nfft // 2 + 1):
            raise AssertionError

        # compute features
        mfccs = mfcc(sig=sig,
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
                     fbanks=mel_filbanks)

        # assert number of returned cepstrum coefficients
        if not mfccs.shape[1] == num_ceps:
            raise AssertionError

        # compute features
        mfccs = mfcc(sig=sig,
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
        if not mfccs.shape[1] == num_ceps:
            raise AssertionError

        # check use energy
        if use_energy:
            mfccs_energy = mfccs[:, 0]
            xfccs_energy = mfcc(sig=sig,
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

            np.testing.assert_almost_equal(mfccs_energy, xfccs_energy, 3)

        # check normalize
        if normalize:
            np.testing.assert_almost_equal(
                mfccs,
                cmvn(
                    cms(
                        mfcc(sig=sig,
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
                    mfccs,
                    lifter_ceps(
                        mfcc(sig=sig,
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
def test_imfcc(sig, fs, num_ceps, pre_emph, nfilts, nfft, low_freq, high_freq,
               scale, dct_type, use_energy, lifter, normalize):
    """
    test IMFCC features module for the following:
        - check if ParameterErrors are raised for:
                - nfilts < num_ceps
                - negative low_freq value
                - high_freq > fs / 2
        - check that the returned number of cepstrums is correct.
        - check the use energy functionality.
        - check normalization.
        - check liftering.
    """
    # check ParameterErrors
    if (low_freq < 0) or (high_freq > fs / 2):
        with pytest.raises(ParameterError):
            # compute the inverse mel filterbanks
            imel_fbanks = mel_fbanks.inverse_mel_filter_banks(nfilts=nfilts,
                                                                nfft=nfft,
                                                                fs=fs,
                                                                low_freq=low_freq,
                                                                high_freq=high_freq,
                                                                scale=scale)

    if (low_freq < 0) or (high_freq > fs / 2) or (nfilts < num_ceps) :
        with pytest.raises(ParameterError):
            imfccs = imfcc(sig=sig,
                           fs=fs,
                           num_ceps=num_ceps,
                           nfilts=nfilts,
                           nfft=nfft,
                           low_freq=low_freq,
                           high_freq=high_freq)
    else:
        imel_fbanks = mel_fbanks.inverse_mel_filter_banks(nfilts=nfilts,
                                                          nfft=nfft,
                                                          fs=fs,
                                                          low_freq=low_freq,
                                                          high_freq=high_freq,
                                                          scale=scale)

        # assert that the filterbank shape is correct given nfilts and nfft
        if not imel_fbanks.shape == (nfilts, nfft // 2 + 1):
            raise AssertionError

        # compute features
        imfccs = imfcc(sig=sig,
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
                       fbanks=imel_fbanks)

        # assert number of returned cepstrum coefficients
        if not imfccs.shape[1] == num_ceps:
            raise AssertionError

        # compute features
        imfccs = imfcc(sig=sig,
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
        if not imfccs.shape[1] == num_ceps:
            raise AssertionError

        # check use energy
        if use_energy:
            imfccs_energy = imfccs[:, 0]
            xfccs_energy = imfcc(sig=sig,
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

            np.testing.assert_almost_equal(imfccs_energy, xfccs_energy, 3)

        # check normalize
        if normalize:
            np.testing.assert_almost_equal(
                imfccs,
                cmvn(
                    cms(
                        imfcc(sig=sig,
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
            # TO FIX: check lifter
            if lifter > 0:
                np.testing.assert_almost_equal(
                    imfccs,
                    lifter_ceps(
                        imfcc(sig=sig,
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
