# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:29:17 2019

@author: amarmore
"""

# Defining current plotting functions.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import musicae.autosimilarity_segmentation as as_seg
from sklearn.decomposition import PCA

def plot_me_this_spectrogram(spec, title = "Spectrogram", x_axis = "x_axis", y_axis = "y_axis", invert_y_axis = True, cmap = cm.Greys, figsize = None, norm = None, vmin = None, vmax = None):
    """
    Plots a spectrogram in a colormesh.
    """
    if figsize != None:
        plt.figure(figsize=figsize)
    elif spec.shape[0] == spec.shape[1]:
        plt.figure(figsize=(7,7))
    padded_spec = pad_factor(spec)
    plt.pcolormesh(np.arange(padded_spec.shape[1]), np.arange(padded_spec.shape[0]), padded_spec, cmap=cmap, norm = norm, vmin = vmin, vmax = vmax)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if invert_y_axis:
        plt.gca().invert_yaxis()
    plt.show()
    
def pad_factor(factor):
    """
    Pads the factor with zeroes on both dimension.
    This is made because colormesh plots values as intervals (post and intervals problem),
    and so discards the last value.
    """
    padded = np.zeros((factor.shape[0] + 1, factor.shape[1] + 1))
    for i in range(factor.shape[0]):
        for j in range(factor.shape[1]):
            padded[i,j] = factor[i,j]
    return np.array(padded)

def plot_latent_space(latent_vectors, labels = None):
    """
    Visualization of the latent projection, as the matrix of representation, and as both autosimilarity and PCA of latent vectors.

    Parameters
    ----------
    latent_vectors : array
        Concatenation of the latent vectors, or matrix of latent representations. 
        (same mathematical meaning but can be of different computation types.)
    labels : None or array, optional
        If labels are set, they will be used to color the output of PCA projection.
        If they are set to None, no label is used. The default is None.

    Returns
    -------
    None, but plots latent visualizations.

    """
    np_lv = np.array(latent_vectors)
    plot_me_this_spectrogram(np_lv.T, figsize=(np_lv.shape[0]/5,np_lv.shape[1]/5), title = "z matrix", x_axis = "Bar index", y_axis = "Latent space")
    
    fig, axs = plt.subplots(1, 2, figsize=(15,7))

    autosimil = as_seg.get_autosimilarity(latent_vectors, transpose = True, normalize = True)
    padded_autosimil = pad_factor(autosimil)
    axs[0].pcolormesh(np.arange(padded_autosimil.shape[1]), np.arange(padded_autosimil.shape[0]), padded_autosimil, cmap = cm.Greys)
    axs[0].set_title('Autosimilarity of the z (projection in latent space)')
    axs[0].invert_yaxis()
    axs[0].set_xlabel("Bar index")
    axs[0].set_ylabel("Bar index")
    
    if np_lv.shape[1] == 2:
        if not isinstance(labels,np.ndarray) and labels == None:
            axs[1].scatter(np_lv[:,0],np_lv[:,1])
        else:
            axs[1].scatter(np_lv[:,0],np_lv[:,1], c=labels)
    else:
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(np_lv)
        if not isinstance(labels,np.ndarray) and labels == None:
            axs[1].scatter(principalComponents[:,0],principalComponents[:,1])
        else:
            axs[1].scatter(principalComponents[:,0],principalComponents[:,1], c=labels)

    axs[1].set_title('PCA of the z (projection in the latent space)')
    plt.show()

def plot_spec_with_annotations(factor, annotations, color = "yellow", title = None):
    """
    Plots a spectrogram with the segmentation annotation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
    plt.show()
    
def plot_spec_with_annotations_abs_ord(factor, annotations, color = "green", title = None, cmap = cm.gray):
    """
    Plots a spectrogram with the segmentation annotation in both x and y axes.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cmap)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = color)
        plt.plot([0,len(factor)], [x,x], '-', linewidth=1, color = color)
    plt.show()

def plot_spec_with_annotations_and_prediction(factor, annotations, predicted_ends, title = "Title"):
    """
    Plots a spectrogram with the segmentation annotation and the estimated segmentation.
    """
    if factor.shape[0] == factor.shape[1]:
        plt.figure(figsize=(7,7))
    plt.title(title)
    padded_fac = pad_factor(factor)
    plt.pcolormesh(np.arange(padded_fac.shape[1]), np.arange(padded_fac.shape[0]), padded_fac, cmap=cm.Greys)
    plt.gca().invert_yaxis()
    for x in annotations:
        plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "#8080FF")
    for x in predicted_ends:
        if x in annotations:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "green")#"#17becf")
        else:
            plt.plot([x,x], [0,len(factor)], '-', linewidth=1, color = "orange")
    plt.show()
