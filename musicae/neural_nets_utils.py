# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:34:29 2021

@author: amarmore
"""

import numpy as np
import musicae.data_manipulation as dm
import musicae.autosimilarity_segmentation as as_seg

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tensorly as tl
import matplotlib.cm as cm
import copy
import random
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# %% Tensor-spectrogram definition (careful: different mode organization than for NTD)
def tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision):
    """
    Returns a tensor-spectrogram from the original spectrogram and bars starts and ends.
    Each bar of the tensor-spectrogram will contain the same number of frames, define by the "subdivision" parameter.
    These frames are selected from an oversampled spectrogram, adapting to the specific size of each bar.

    Parameters
    ----------
    spectrogram : list of list of floats or numpy array
        The spectrogram to return as a tensor-spectrogram.
    bars : list of tuples
        List of the bars (start, end), in seconds, to cut the spectrogram at bar delimitation.
    hop_length_seconds : float
        The hop_length, in seconds.
    subdivision : integer
        The number of subdivision of the bar to be contained in each slice of the tensor.

    Returns
    -------
    np.array tensor
        The tensor-spectrogram as a np.array.

    """
    barwise_spec = []
    bars_idx = dm.segments_from_time_to_frame_idx(bars[1:], hop_length_seconds)
    for idx, beats in enumerate(bars_idx):
        t_0 = beats[0]
        t_1 = beats[1]
        samples = [int(round(t_0 + k * (t_1 - t_0)/subdivision)) for k in range(subdivision)]
        if samples[-1] < spectrogram.shape[1]:
            barwise_spec.append(spectrogram[:,samples])
    return np.array(barwise_spec)

# %% Dataloaders generation
def generate_dataloader(tensor_barwise):
    """
    Generates a torch.DataLoader from the tensor spectrogram.

    Parameters
    ----------
    tensor_barwise : np.array tensor
        The tensor-spectrogram as a np.array.

    Returns
    -------
    data_loader : torch.DataLoader
        torch.DataLoader for this song and this tensor-spectrogram.
    """
    batch_size = 8
    num_workers = 0
    nb_bars = tensor_barwise.shape[0]
    freq_len, subdivision_bars = tensor_barwise.shape[1], tensor_barwise.shape[2]

    barwise_spec = copy.deepcopy(tensor_barwise)
    
    barwise_spec = barwise_spec.reshape((nb_bars, 1, freq_len, subdivision_bars))
    data_loader = torch.utils.data.DataLoader(barwise_spec, batch_size=batch_size, num_workers=num_workers)
    
    # flatten_barwise_spec = barwise_spec.reshape((nb_bars, freq_len*subdivision_bars), order="F")
    # flatten_simplet_data_loader = torch.utils.data.DataLoader(flatten_barwise_spec, batch_size=batch_size, num_workers=num_workers)
    return data_loader

def generate_triplet_dataloader(tensor_barwise, top_partition = 0.1, medium_partition = 0.75):
    """
    Generates torch.DataLoaders for TripletLoss, with a positive and a negative example for each bar.
    Positive and negative examples are randomly selected from most similar and less similar bars.
    The similarity is the feature_wise similarity.
    Both arguments ``top_partition'' and ``medium_partition'' are dedicated to thresholding most and less similar for random selection.

    Parameters
    ----------
    tensor_barwise : np.array tensor
        The tensor-spectrogram as a np.array.
    top_partition : float \in [0,1], optional
        Percentage of most similar bars (in feature-wise similarity) on which select a positive example.
        The default is 0.1, corresponding to top-10% of similar bars.
    medium_partition : float \in [0,1], optional
        Percentage of less similar bars (in feature-wise similarity) on which select a positive example.
        The default is 0.75, selecting the bottom-75% of similar bars.

    Returns
    -------
    triplet_data_loader : torch.DataLoader
        torch.DataLoader for this song and this tensor-spectrogram.
        Each barwise spectrogram is kept as a matrix, for convolutional networks.
    flatten_triplet_data_loader : torch.DataLoader
        torch.DataLoader for this song and this tensor-spectrogram.
        On this DataLoader, each barwise spectrogram is flattened, for networks which needs vectors as inputs.

    """
    batch_size = 8
    num_workers = 0
    signal_autosimilarity = as_seg.get_autosimilarity(tl.unfold(tensor_barwise,0), transpose = True, normalize = True)
    nb_bars = tensor_barwise.shape[0]
    vectorized_spec_dim = tensor_barwise.shape[1]*tensor_barwise.shape[2]
    
    barwise_spec = copy.deepcopy(tensor_barwise)
    triplet_data = []
    flatten_triplet_data = []
    
    threshed_mat = np.ones((nb_bars, nb_bars))

    high_thresh = int(top_partition * nb_bars) + 1
    medium_thresh = int(medium_partition * nb_bars)

    for index, bar in enumerate(barwise_spec):
        this_bar_similarities = signal_autosimilarity[index]
        highest_indexes = np.argpartition(this_bar_similarities, -high_thresh)[-high_thresh:]
        threshed_mat[index,highest_indexes] = 2
        threshed_mat[highest_indexes,index] = 2
        selected_high = random.choice(highest_indexes)
        while selected_high == index:
            selected_high = random.choice(highest_indexes)
        positive_bar = barwise_spec[selected_high]

        lowest_indexes = np.argpartition(this_bar_similarities, medium_thresh)[:medium_thresh]
        threshed_mat[index,lowest_indexes] = 0
        threshed_mat[lowest_indexes,index] = 0
        selected_low = random.choice(lowest_indexes)
        negative_bar = barwise_spec[selected_low]

        triplet_data.append((bar, positive_bar, negative_bar))
        flatten_triplet_data.append((bar.reshape((vectorized_spec_dim), order="F"), positive_bar.reshape((vectorized_spec_dim), order="F"), negative_bar.reshape((vectorized_spec_dim), order="F")))

    triplet_data_loader = torch.utils.data.DataLoader(triplet_data, batch_size=batch_size, num_workers=num_workers)
    flatten_triplet_data_loader = torch.utils.data.DataLoader(flatten_triplet_data, batch_size=batch_size, num_workers=num_workers)
    return triplet_data_loader, flatten_triplet_data_loader
    
# %% Deterministic initializations for networks
def seeded_weights_init_kaiming(m): 
    """
    Determinstic initialization of weights with Kaiming uniform distribution.

    Parameters
    ----------
    m : torch.nn
        A layer of the network.
    """       
    # if isinstance(m, nn.BatchNorm1d):
    #     nn.init.zeros_(m.bias)
    #     nn.init.ones_(m.weight)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.manual_seed(42)
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
def seeded_weights_init_xavier(m):
    """
    Determinstic initialization of weights with Xavier uniform distribution.

    Parameters
    ----------
    m : torch.nn
        A layer of the network.
    """    
    # if isinstance(m, nn.BatchNorm1d):
    #     nn.init.zeros_(m.bias)
    #     nn.init.ones_(m.weight)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.manual_seed(42)
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        
def weights_init_sparse(m):
    """
    Determinstic initialization of weights with sparse initialization.

    Parameters
    ----------
    m : torch.nn
        A layer of the network.
    """   
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        torch.nn.init.sparse_(m.weight, sparsity = 0.1)
        
def weights_ones(m):
    """
    Determinstic initialization of weights with layers full of ones.

    Parameters
    ----------
    m : torch.nn
        A layer of the network.
    """   
    if isinstance(m, nn.Linear):# or isinstance(m, nn.ConvTranspose2d):
        nn.init.ones_(m.weight) 
        
# %% Triplet losees
class TripletLoss(nn.Module):
    """
    Triplet Loss class, following the Triplet Loss paradigm. See [1] for details.
        
    Comes from: https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
    
    References
    ----------
    [1] Ho, K., Keuper, J., Pfreundt, F. J., & Keuper, M. (2021, January). 
    Learning embeddings for image clustering: An empirical study of triplet loss approaches. 
    In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 87-94). IEEE.
    """
    
    def __init__(self, margin=1.0):
        """
        Constructor of the loss.

        Parameters
        ----------
        margin : float, optional
            Margin for the triplet loss. The default is 1.0.

        Returns
        -------
        None.

        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    
class TripletLossDoubleMargin(nn.Module):
    """
    Triplet Loss with positive and negative margins, following the work of [1]
    
    References
    ----------
    [1] Ho, K., Keuper, J., Pfreundt, F. J., & Keuper, M. (2021, January). 
    Learning embeddings for image clustering: An empirical study of triplet loss approaches. 
    In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 87-94). IEEE.
    """
    def __init__(self, pos_margin=1.0, neg_margin = 3.0):
        """
        Constructor of the loss.

        Parameters
        ----------
        pos_margin : float, optional
            Margin for positive examples. The default is 1.0.
        neg_margin : float, optional
            Margin for negative examples. The default is 3.0.

        Returns
        -------
        None.

        """
        super(TripletLossDoubleMargin, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(self.neg_margin - distance_negative) + torch.relu(distance_positive - self.pos_margin)
        return losses.mean()