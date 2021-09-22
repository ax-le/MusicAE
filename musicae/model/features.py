# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:54:59 2020

@author: amarmore
"""

import numpy as np
import librosa.core
import librosa.feature
import librosa.effects
from math import inf
import soundfile as sf
import musicae.model.errors as err

def get_spectrogram(signal, sr, feature, hop_length, n_fft = 2048, fmin = 98, n_mfcc = 20):
    """
    Returns a spectrogram, from the signal of a song.
    Different types of spectrogram can be computed, which are specified by the argument "feature".
    All these spectrograms are computed with the toolbox librosa [1].
    
    Parameters
    ----------
    signal : numpy array
        Signal of the song.
    sr : float
        Sampling rate of the signal, (typically 44100Hz).
    feature : String
        The types of spectrograms to compute.
            - stft : computes the Short-Time Fourier Transform of the signal.
            Returns the Power spectrogram.
            - pcp : computes a chromagram.
            NB: this chromagram has been specificly fitted as a team, 
            and the arguments are non standard but rather technical choices.
            - pcp_stft : computes a chromagram from the stft of the song.
            - cqt : computes a Constant-Q transform of the song.
            - log_cqt : computes the logarithm of the Constant-Q transform of the song.
            - tonnetz : computes the tonnetz representation of the song.
            - pcp_tonnetz : computes the tonnetz representation of the song, starting from the chromas.
                It allows us to better control paramaters over the computation of tonnetz, 
                and can reduce computation when chromas are already computed (for scripts loading already computed spectrograms).
            - mfcc : computes the Mel-Frequency Cepstral Coefficients of the song.
            - mel_grill : computes the mel-spectrogram of the song, as dimensioned by [2].
            - log_mel_grill : computes the logarithm of the previously defined mel-spectrogram.
            - pos_log_mel_grill : computes the log(mel + 1) of the previously defined mel-spectrogram.

    hop_length : integer
        The desired hop_length, which is the step between two frames (ie the time "discretization" step)
        It is expressed in terms of number of samples, which are defined by the sampling rate.
    n_fft : integer, optional
        Number of frames by stft feature.
        The default is 2048.
    fmin : integer, optional
        The minimal frequence to consider, used for denoising.
        The default is 98.
    n_mfcc : integer, optional
        Number of mfcc features.
        The default is 20 (as in librosa).

    Raises
    ------
    InvalidArgumentValueException
        If the "feature" argument is not presented above.

    Returns
    -------
    numpy array
        Spectrogram of the signal.
        
    References
    ----------
    [1] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015, July).
    librosa: Audio and music signal analysis in python. 
    In Proceedings of the 14th python in science conference (Vol. 8).
    
    [2] Grill, T., & Schlüter, J. (2015, October). 
    Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations. 
    In ISMIR (pp. 531-537).
    """
    if feature.lower() == "stft":
        if len(signal.shape) == 1:
            stft = librosa.core.stft(np.asfortranarray(signal), n_fft=n_fft, hop_length = hop_length)
            power_spectrogram = np.abs(stft) ** 2
            return power_spectrogram
        
        power_spectrogram = np.abs(librosa.core.stft(np.asfortranarray(signal[:,0]), n_fft=n_fft, hop_length = hop_length))**2
        for i in range(1,signal.shape[1]):
            power_spectrogram += np.abs(librosa.core.stft(np.asfortranarray(signal[:,i]), n_fft=n_fft, hop_length = hop_length))**2
        return power_spectrogram
    
    elif feature.lower() == "pcp_stft":
        if len(signal.shape) == 1:
            audio_harmonic, _ = librosa.effects.hpss(y=np.asfortranarray(signal))
            chroma_stft = librosa.feature.chroma_stft(y=audio_harmonic, sr=sr, n_fft = n_fft, hop_length=hop_length)
            return chroma_stft
        audio_harmonic, _ = librosa.effects.hpss(y=np.asfortranarray(signal[:,0]))
        chroma_stft = librosa.feature.chroma_stft(y=audio_harmonic, sr=sr, n_fft = n_fft, hop_length=hop_length)
        for i in range(1,signal.shape[1]):
            audio_harmonic, _ = librosa.effects.hpss(y=np.asfortranarray(signal[:,i]))
            chroma_stft += librosa.feature.chroma_stft(y=audio_harmonic, sr=sr, n_fft = n_fft, hop_length=hop_length)   
        return chroma_stft
    elif feature == "pcp":
        norm=inf # Columns normalization
        win_len_smooth=82 # Size of the smoothign window
        n_octaves=6
        bins_per_chroma = 3
        bins_per_octave=bins_per_chroma * 12
        if len(signal.shape) == 1:
            return librosa.feature.chroma_cens(y=np.asfortranarray(signal),sr=sr,hop_length=hop_length,
                                   fmin=fmin, n_chroma=12, n_octaves=n_octaves, bins_per_octave=bins_per_octave,
                                   norm=norm, win_len_smooth=win_len_smooth)
        
        pcp = librosa.feature.chroma_cens(y=np.asfortranarray(signal[:,0]),sr=sr,hop_length=hop_length,
                                   fmin=fmin, n_chroma=12, n_octaves=n_octaves, bins_per_octave=bins_per_octave,
                                   norm=norm, win_len_smooth=win_len_smooth)
        for i in range(1,signal.shape[1]):
            pcp += librosa.feature.chroma_cens(y=np.asfortranarray(signal[:,i]),sr=sr,hop_length=hop_length,
                                   fmin=fmin, n_chroma=12, n_octaves=n_octaves, bins_per_octave=bins_per_octave,
                                   norm=norm, win_len_smooth=win_len_smooth)
    
        return pcp
    elif feature.lower() == "cqt":
        if len(signal.shape) == 1:
            constant_q_transf = librosa.core.cqt(np.asfortranarray(signal), sr = sr, hop_length = hop_length)
            power_cqt = np.abs(constant_q_transf) ** 2
            return power_cqt
        power_cqt = np.abs(librosa.core.cqt(np.asfortranarray(signal[:,0]), sr = sr, hop_length = hop_length)) ** 2
        for i in range(1,signal.shape[1]):
            power_cqt += np.abs(librosa.core.cqt(np.asfortranarray(signal[:,i]), sr = sr, hop_length = hop_length)) ** 2
        return power_cqt
    elif feature.lower() == "log_cqt":
        if len(signal.shape) == 1:
            constant_q_transf = librosa.core.cqt(np.asfortranarray(signal), sr = sr, hop_length = hop_length)
            power_cqt = np.abs(constant_q_transf) ** 2
            log_cqt = ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(np.array(power_cqt)), ref=np.max)) + 1.0
            return log_cqt
        power_cqt = np.abs(librosa.core.cqt(np.asfortranarray(signal[:,0]), sr = sr, hop_length = hop_length)) ** 2
        for i in range(1,signal.shape[1]):
            power_cqt += np.abs(librosa.core.cqt(np.asfortranarray(signal[:,i]), sr = sr, hop_length = hop_length)) ** 2
        log_cqt = ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(np.array(power_cqt)), ref=np.max)) + 1.0
        return log_cqt
    elif feature.lower() == "tonnetz":
        if len(signal.shape) == 1:
            return librosa.feature.tonnetz(np.asfortranarray(signal), sr = sr)
        tonnetz = librosa.feature.tonnetz(np.asfortranarray(signal[:,0]), sr = sr)
        for i in range(1,signal.shape[1]):
            tonnetz += librosa.feature.tonnetz(np.asfortranarray(signal[:,i]), sr = sr)
        return tonnetz
    elif feature.lower() == "pcp_tonnetz":
        return librosa.feature.tonnetz(y=None, sr = None, chroma = get_spectrogram(signal, sr, "pcp", hop_length, fmin = fmin))
    # elif feature.lower() == "hcqt":
    #     return my_compute_hcqt(np.asfortranarray(signal[:,0]), sr)
    
    elif feature.lower() == "mfcc":
        if len(signal.shape) == 1:
            return librosa.feature.mfcc(np.asfortranarray(signal), sr = sr, hop_length = hop_length, n_mfcc=n_mfcc)
        mfcc = librosa.feature.mfcc(np.asfortranarray(signal[:,0]), sr = sr, hop_length = hop_length, n_mfcc=n_mfcc)
        for i in range(1,signal.shape[1]):
            mfcc += librosa.feature.mfcc(np.asfortranarray(signal[:,i]), sr = sr, hop_length = hop_length, n_mfcc=n_mfcc)
        return mfcc
    
    # For Mel spectrograms, we use the same parameters as the ones of [2].
    # [2] Grill, Thomas, and Jan Schlüter. "Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations." ISMIR. 2015.
    elif feature.lower() == "mel_grill":
        if len(signal.shape) == 1:
            return np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        mel = np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,0]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        for i in range(1,signal.shape[1]):
            mel += np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,i]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        return mel
    
    elif feature == "log_mel_grill":
        if len(signal.shape) == 1:
            return librosa.power_to_db(np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000)))
        mel = np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,0]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        for i in range(1,signal.shape[1]):
            mel += np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,i]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        return librosa.power_to_db(mel)
    
    elif feature == "pos_log_mel_grill":
        if len(signal.shape) == 1:
            mel = np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
            return librosa.power_to_db(mel + np.ones(mel.shape))
        mel = np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,0]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        for i in range(1,signal.shape[1]):
            mel += np.abs(librosa.feature.melspectrogram(np.asfortranarray(signal[:,i]), sr = sr, n_fft=2048, hop_length = hop_length, n_mels=80, fmin=80.0, fmax=16000))
        return librosa.power_to_db(mel + np.ones(mel.shape))
    
    elif feature == "mel" or feature == "log_mel" or feature == "pos_log_mel":
        raise err.InvalidArgumentValueException("Invalid feature parameter, aren't you looking for mel_grill/log_mel_grill (the only available Mel Spectrograms)?")
    else:
        raise err.InvalidArgumentValueException(f"Unknown signal representation: {feature}.")
