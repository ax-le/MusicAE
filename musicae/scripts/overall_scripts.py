# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:34:39 2020

@author: amarmore
"""

# A module containing some high-level scripts for decomposition and/or segmentation.

import soundfile as sf
import librosa.core
import librosa.feature
import os
import numpy as np

import musicae.data_manipulation as dm
import musicae.convolutional as convolutional
import musicae.neural_nets_utils as nn_utils
import musicae.model.features as features
import musicae.model.errors as err

def load_RWC_dataset(music_folder_path, annotations_type = "MIREX10"):
    """
    Load the data on the RWC dataset, ie path of songs and annotations.
    The annotations can be either AIST or MIREX 10.

    Parameters
    ----------
    music_folder_path : String
        Path of the folder to parse.
    annotations_type : "AIST" [1] or "MIREX10" [2]
        The type of annotations to load (both have a specific behavior and formatting)
        The default is "MIREX10"

    Raises
    ------
    NotImplementedError
        If the format is not taken in account.

    Returns
    -------
    numpy array
        list of list of paths, each sublist being of the form [song, annotations, downbeat(if specified)].
        
    References
    ----------
    [1] Goto, M. (2006, October). AIST Annotation for the RWC Music Database. In ISMIR (pp. 359-360).
    
    [2] Bimbot, F., Sargent, G., Deruty, E., Guichaoua, C., & Vincent, E. (2014, January). 
    Semiotic description of music structure: An introduction to the Quaero/Metiss structural annotations.

    """
    # Load dataset paths at the format "song, annotations, downbeats"
    paths = []
    for file in os.listdir(music_folder_path):
        if file[-4:] == ".wav":
            file_number = "{:03d}".format(int(file[:-4]))
            ann = dm.get_annotation_name_from_song(file_number, annotations_type)
            paths.append([file, ann])
    return np.array(paths)

# %% Loading or persisting bars, spectrograms and already computed neural networks latent projections
def load_or_save_bars(persisted_path, song_path):
    """
    Computes the bars for this song, or load them if they were already computed.

    Parameters
    ----------
    persisted_path : string
        Path where the bars should be found.
    song_path : string
        The path of the signal of the song.

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        bars = np.load("{}/bars/{}.npy".format(persisted_path, song_name))
    except:
        bars = dm.get_bars_from_audio(song_path)
        np.save("{}/bars/{}".format(persisted_path, song_name), bars)
    return bars

def load_bars(persisted_path, song_name):
    """
    Loads the bars for this song, which were persisted after a first computation.

    Parameters
    ----------
    persisted_path : string
        Path where the bars should be found.
    song_name : string
        Name of the song (identifier of the bars to load).

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    """
    raise err.OutdatedBehaviorException("You should use load_or_save_bars(persisted_path, song_path) instead, as it handle the fact that bars weren't computed yet.")
    bars = np.load("{}/bars/{}.npy".format(persisted_path, song_name))
    return bars
    
def load_or_save_spectrogram(persisted_path, song_path, feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20):
    """
    Computes the spectrogram for this song, or load it if it were already computed.

    Parameters
    ----------
    persisted_path : string
        Path where the spectrogram should be found.
    song_path : string
        The path of the signal of the song.
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.

    Returns
    -------=
    spectrogram : numpy array
        The pre-computed spectorgram.
    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        if "stft" in feature:   
            if "nfft" not in feature:
                spectrogram = np.load("{}/spectrograms/{}_{}-nfft{}_stereo_{}.npy".format(persisted_path, song_name, feature, n_fft, hop_length))
            else:
                spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}.npy".format(persisted_path, song_name, feature, hop_length))
        elif feature == "mel" or feature == "log_mel":
            raise err.InvalidArgumentValueException("Invalid mel parameter, are't you looking for mel_grill?")
        elif "mfcc" in feature:
            if "nmfcc" not in feature:
                spectrogram = np.load("{}/spectrograms/{}_{}-nmfcc{}_stereo_{}.npy".format(persisted_path, song_name, feature, n_mfcc, hop_length))
            else:
                spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}.npy".format(persisted_path, song_name, feature, hop_length))
        elif feature == "pcp":
            spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}_{}.npy".format(persisted_path, song_name, feature, hop_length, fmin))
        else:
            spectrogram = np.load("{}/spectrograms/{}_{}_stereo_{}.npy".format(persisted_path, song_name, feature, hop_length))

    except FileNotFoundError:
        the_signal, original_sampling_rate = sf.read(song_path)
        if original_sampling_rate != 44100:
            the_signal = librosa.core.resample(np.asfortranarray(the_signal), original_sampling_rate, 44100)
        if "stft" in feature:
            if "nfft" not in feature: 
                spectrogram = features.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft)
                np.save("{}/spectrograms/{}_{}-nfft{}_stereo_{}".format(persisted_path, song_name, feature, n_fft, hop_length), spectrogram)
                return spectrogram
            else:              
                n_fft_arg = int(feature.split("nfft")[1])
                spectrogram = features.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft_arg)
                np.save("{}/spectrograms/{}_{}_stereo_{}".format(persisted_path, song_name, feature, hop_length), spectrogram)
                return spectrogram
        if feature == "mel" or feature == "log_mel":
            raise err.InvalidArgumentValueException("Invalid mel parameter, are't you looking for mel_grill?")
        if "mfcc" in feature:
            if "nmfcc" not in feature:
                spectrogram = features.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc)
                np.save("{}/spectrograms/{}_{}-nmfcc{}_stereo_{}".format(persisted_path, song_name, feature, n_mfcc, hop_length), spectrogram)
                return spectrogram        
            else:
                n_mfcc_arg = int(feature.split("nmfcc")[1])
                spectrogram = features.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc_arg)
                np.save("{}/spectrograms/{}_{}_stereo_{}".format(persisted_path, song_name, feature, hop_length), spectrogram)
                return spectrogram
        if feature == "pcp_tonnetz":
            # If chromas are already computed, try to load them instead of recomputing them.
            chromas = load_or_save_spectrogram(persisted_path, song_path, "pcp", hop_length, fmin = fmin)
            spectrogram = librosa.feature.tonnetz(y=None, sr = None, chroma = chromas)
            np.save("{}/spectrograms/{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin), spectrogram)
            return spectrogram
        # if feature == "tonnetz":
        #     hop_length = "fixed"
        #     fmin = "fixed"
        if feature == "pcp":
            # If it wasn't pcp_tonnetz, compute the spectrogram, and then save it.
            spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length, fmin = fmin)
            np.save("{}/spectrograms/{}_{}_stereo_{}_{}".format(persisted_path, song_name, feature, hop_length, fmin), spectrogram)
            return spectrogram
        
        spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length)
        np.save("{}/spectrograms/{}_{}_stereo_{}".format(persisted_path, song_name, feature, hop_length), spectrogram)
        return spectrogram

    return spectrogram

def load_or_save_spectrogram_and_bars(persisted_path, song_path, feature, hop_length, fmin = 98, n_fft = 2048, n_mfcc = 20):
    """
    Loads the spectrogram and the bars for this song, which were persisted after a first computation, or compute them if they weren't found.

    Parameters
    ----------
    persisted_path : string
        Path where the bars and the spectrogram should be found.
    song_path : string
        The path of the signal of the song.
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.
    n_fft and n_mfcc : integers, optional
        Both arguments are used respectively for the stft and for the mfcc computation, and are used to 

    Returns
    -------
    bars : list of tuple of floats
        The persisted bars for this song.
    spectrogram : numpy array
        The pre-computed spectorgram.
    """
    bars = load_or_save_bars(persisted_path, song_path)
    spectrogram = load_or_save_spectrogram(persisted_path, song_path, feature, hop_length, fmin = fmin, n_fft = n_fft, n_mfcc = n_mfcc)
    return bars, spectrogram

def load_or_save_tensor_spectrogram(persisted_path, song_path, feature, hop_length, subdivision_bars, fmin = 98, n_fft = 2048, n_mfcc = 20):
    """
    Loads the tensor_pectrogram for this song, which was persisted after a first computation, or compute it if it wasn't found.

    You should prefer load_or_save_spectrogram, as it allows more possibility about tensor folding, except if you're short in space on your disk.

    Parameters
    ----------
    persisted_path : string
        Path where the bars and the spectrogram should be found.
    song_path : string
        The path of the signal of the song.
    feature : string
        Feature of the spectrogram, part of the identifier of the spectrogram.
    hop_length : integer
        hop_length of the spectrogram, part of the identifier of the spectrogram.
    fmin : integer
        Minimal frequence for the spectrogram, part of the identifier of the spectrogram.
        The default is 98.
    n_fft and n_mfcc : integers, optional
        Both arguments are used respectively for the stft and for the mfcc computation, and are used to 

    Returns
    -------
    numpy array
        The tensor spectrogram of this song.

    """
    song_name = song_path.split("/")[-1].replace(".wav","").replace(".mp3","")
    try:
        tensor_barwise = np.load(f"{persisted_path}/tensor_barwise_ae/{song_name}_{feature}_hop{hop_length}_subdiv{subdivision_bars}.npy", allow_pickle = True)
        return tensor_barwise
    except FileNotFoundError:
        raise NotImplementedError("YOlo")
        the_signal, original_sampling_rate = sf.read(song_path)
        if original_sampling_rate != 44100:
            the_signal = librosa.core.resample(np.asfortranarray(the_signal), original_sampling_rate, 44100)
        if "stft" in feature and "nfft" in feature:
            if "nfft" not in feature: 
                spectrogram = features.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft)
            else:              
                n_fft_arg = int(feature.split("nfft")[1])
                spectrogram = features.get_spectrogram(the_signal, 44100, "stft", hop_length, n_fft = n_fft_arg)
        elif "mfcc" in feature:
            if "nmfcc" not in feature:
                spectrogram = features.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc)
            else:
                n_mfcc_arg = int(feature.split("nmfcc")[1])
                spectrogram = features.get_spectrogram(the_signal, 44100, "mfcc", hop_length, n_mfcc = n_mfcc_arg)
        else:
            spectrogram = features.get_spectrogram(the_signal, 44100, feature, hop_length, fmin = fmin)
            
        hop_length_seconds = hop_length/44100
        bars = load_or_save_bars(persisted_path, song_path)
        tensor_spectrogram = nn_utils.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision_bars)
        np.save(f"{persisted_path}/tensor_barwise_ae/{song_name}_{feature}_hop{hop_length}_subdiv{subdivision_bars}", tensor_spectrogram)
        return tensor_spectrogram
    

def load_or_save_convolutional_projection(persisted_path, song_name, data_loader, dim_latent_space, sparsity_lambda, nn = False, lr = 1e-3, n_epochs = 1000, norm_tensor = None, feature = "pcp", hop_length = 32, subdivision_bars = 96, freq_len = 12, compute_if_not_persisted = True):
    """
    Loads the neural network projection for this song, which was persisted after its computation.

    Parameters
    ----------
    persisted_path : string
        Path of the folder where the projections should be found.
    song_name : int or string
        Identifier of the song.
    data_loader : torch.DataLoader
        The DataLoader associated with the barwise tensor of this song.
    dim_latent_space : int
        Dimension of the latent space.
    sparsity_lambda : None or float
        The sparisty ponderation parameter.
        If set to None, no sparsity is enforced.
    nn : boolean, optional
        DEPRECATED. Whether the latent vectors should be nonnegative or not. 
        Accepted behavior nowadays is not enforcing nonnegativity. The default is False.
    lr : float, optional
        Learning rate of the network. The default is 1e-3.
    n_epochs : int, optional
        Number of epochs to perform. The default is 1000.
    norm_tensor : float, optional
        Norm of the barwise tensor. Used to noramlize the sparsity parameter, so useless when sparsity lambda is set to None.
        The default is None.
    feature : string, optional
        The feature used to represent the song. See model.feature.py for details. The default is "pcp".
    hop_length : int, optional
        Hop_length used to compute the original spectrogram. The default is 32.
    subdivision_bars : int, optional
        The number of subdivision of the bar to be contained in each slice of the tensor.
    freq_len : int, optional
        Dimension of the frequency mode (frequency-related representation of music). The default is 12.
    compute_if_not_persisted : boolean, optional
        Indicating hether the network should be computed if it's not found in persisted networks.
        Should be set to True if networks are computed for the first time, and to False to speed up computation in tests where they have already been computed.
        The default is True.

    Raises
    ------
    FileNotFoundError
        If the network asn't found and if ``FileNotFoundError'' is set to False.

    Returns
    -------
    projection : numpy array
        Latent projection of each bar through this network.

    """
    if nn:
        raise err.OutdatedBehaviorException("So-called ``nonnegative networks'' are not supported anymore, as they weren't succesful.") from None
    persisted_params = "song{}_feature{}_hop{}_subdivbars{}_initkaiming_lr{}_nepochs{}".format(song_name, feature, hop_length, subdivision_bars, lr, n_epochs)
    
    conv_save_name = "{}/neural_nets/conv_4_16_k3_transk3_latentfc{}_{}".format(persisted_path, dim_latent_space, persisted_params)
    conv_load_name = "{}/neural_nets/transfered/conv_4_16_k3_transk3_latentfc{}_{}".format(persisted_path, dim_latent_space, persisted_params)
    
    if sparsity_lambda == None:
        try:
            projection = np.load("{}.npy".format(conv_load_name), allow_pickle = True)
        except FileNotFoundError:
            try:
                projection = np.load("{}.npy".format(conv_save_name), allow_pickle = True)
            except FileNotFoundError:
                if compute_if_not_persisted:
                    conv_model = convolutional.ConvolutionalAutoencoder(input_size_x = subdivision_bars, input_size_y = freq_len, dim_latent_space = dim_latent_space, nonnegative = False)
                    conv_model = conv_model.my_optim_method(n_epochs, data_loader, lr=lr, verbose = False, labels = None)
                    projection = conv_model.get_latent_projection(data_loader)
                    np.save(conv_save_name, projection)
                else:
                    raise FileNotFoundError(f"Neural network projection not found, check the name: {conv_save_name}") from None
        
    # Sparse
    else:

        ## When the sparsity parameter is normalized by the norm of the tensor
        # conv_sparse_load_name = "{}_sparse_lambda_l2_normed{}".format(conv_load_name, sparsity_lambda)
        # conv_sparse_save_name = "{}_sparse_lambda_l2_normed{}".format(conv_save_name, sparsity_lambda)
        
        ## When the sparsity parameter is not normalized
        norm_tensor = 1
        conv_sparse_load_name = "{}_sparse_lambda{}".format(conv_load_name, sparsity_lambda)
        conv_sparse_save_name = "{}_sparse_lambda{}".format(conv_save_name, sparsity_lambda)
        
        try:
            projection = np.load("{}.npy".format(conv_sparse_load_name), allow_pickle = True)
        except FileNotFoundError:
            try:
                projection = np.load("{}.npy".format(conv_sparse_save_name), allow_pickle = True)
            except FileNotFoundError:
                if compute_if_not_persisted:
                    assert norm_tensor != None
                    conv_sparse_model = convolutional.ConvolutionalAutoencoderSparse(input_size_x = subdivision_bars, input_size_y = freq_len, nonnegative = False,
                                                                                     dim_latent_space = dim_latent_space, sparsity_lambda = sparsity_lambda, norm_tensor = norm_tensor)
                    
                    conv_sparse_model = conv_sparse_model.my_optim_method(n_epochs, data_loader, lr=lr, verbose = False, labels = None)
                    projection = conv_sparse_model.get_latent_projection(data_loader)
                    np.save(conv_sparse_save_name, projection)
                else:
                    raise FileNotFoundError(f"Neural network projection not found, check the name: {conv_sparse_save_name}") from None
    return projection