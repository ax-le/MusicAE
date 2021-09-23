# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:12:10 2021

@author: amarmore
"""

import numpy as np
import musicae.model.features
import musicae.data_manipulation as dm
import musicae.autosimilarity_segmentation as as_seg
#import musicae.convolutional as conv
import musicae.neural_nets_utils as nn_utils
import musicae.scripts.overall_scripts as scr
import musicae.scripts.default_path as paths
import musicae.model.errors as err

# from musicae.model.current_plot import *

import mirdata

from IPython.display import display, Markdown
import pandas as pd
pd.set_option('precision', 4)
import warnings
import torch
import math
import tensorly as tl
import random
import os

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# %% Tests on RWC Pop dataset
def final_results_fixed_conditions_rwc(dataset, feature, dim_latent_space, sparsity_lambda, lr = 1e-3, n_epochs = 1000, subdivision = 96, penalty_func = "modulo8", penalty_weight = 1, convolution_type = "mixed", compute_if_not_persisted = False, legend = "in unprecised conditions."):
    """
    Segmentation results on the RWC Pop dataset when the latent space and the sparsity parameter are fixed before computation.
    """
    if dataset == "full":
        dataset_path = paths.path_entire_rwc
    elif dataset == "odd_songs":
        dataset_path = paths.path_odd_songs_rwc
    elif dataset == "even_songs":
        dataset_path = paths.path_even_songs_rwc
    elif dataset == "debug":
        dataset_path = paths.path_debug_rwc
    else:
        raise err.InvalidArgumentValueException(f"Dataset type not understood: {dataset}") from None

    annotations_type = "MIREX10"
    annotations_folder = f"{paths.path_annotation_rwc}/{annotations_type}"
    song_list = scr.load_RWC_dataset(dataset_path, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    subdivision_bars = 96
    zero_five_results = []
    three_results = []
    deviation = []

    for song_and_annotations in song_list:
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}/{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        song_path = f"{dataset_path}/{song_number}.wav"
        
        spectrogram = scr.load_or_save_spectrogram(paths.path_data_persisted_rwc, song_path, feature, hop_length)
    
        bars = scr.load_or_save_bars(paths.path_data_persisted_rwc, song_path)
                
        tensor_barwise = nn_utils.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision_bars)
        data_loader = nn_utils.generate_dataloader(tensor_barwise)
        
        projection_conv = scr.load_or_save_convolutional_projection(paths.path_data_persisted_rwc, song_number, data_loader, dim_latent_space, sparsity_lambda = sparsity_lambda, nn = False, lr = lr, n_epochs = n_epochs, feature = feature, hop_length = hop_length, subdivision_bars = 96, freq_len = 12, compute_if_not_persisted = compute_if_not_persisted)
        projection_conv = np.where(abs(projection_conv) < 1e-15, 0, projection_conv)
        autosimilarity_conv = as_seg.get_autosimilarity(projection_conv, transpose = True, normalize = True)
                    
        segments = as_seg.dynamic_convolution_computation(autosimilarity_conv, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
        segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                    
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
        zero_five_results.append([round(prec,4),round(rap,4),round(f_mes,4)])
    
        prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 3)
        three_results.append([round(prec,4),round(rap,4),round(f_mes,4)])
        
        r_to_e, e_to_r = dm.compute_median_deviation_of_segmentation(references_segments, segments_in_time)
        deviation.append([r_to_e, e_to_r])
    
    results_at_zero_five = np.array([np.mean(np.array(zero_five_results)[:,i]) for i in range(3)])
    results_at_three = np.array([np.mean(np.array(three_results)[:,i]) for i in range(3)])
    line = [results_at_zero_five, results_at_three]
    dataframe = pd.DataFrame(line, columns = ['Precision', 'Recall', 'F measure'], index = ["Results with 0.5 seconds tolerance window {}".format(legend), "Results with 3 seconds tolerance window {}".format(legend)])
    display(dataframe)
    
    # dataframe_zero_five = pd.DataFrame(results_at_zero_five, index = ['Precision', 'Recall', 'F measure'], columns = ["Results with 0.5 seconds tolerance window {}".format(legend)])
    # display(dataframe_zero_five.T)
    # dataframe_three = pd.DataFrame(results_at_three, index = ['Precision', 'Recall', 'F measure'], columns = ["Results with 3 seconds tolerance window {}".format(legend)])
    # display(dataframe_three.T)
    # mean_deviation = np.array([np.mean(np.array(deviation)[:,i]) for i in range(2)])
    # dataframe_deviation = pd.DataFrame(mean_deviation, index = ['Reference to Estimation mean deviation','Estimation to Reference mean deviation'], columns = ["Mean deviation of conv between estimations and references {}".format(legend)])
    # display(dataframe_deviation.T)
    
    return results_at_zero_five, results_at_three

def several_conditions_with_cross_validation_of_param_RWC(learning_dataset, testing_dataset, feature, dim_latent_space_range, sparsity_range, subdivision = 96, penalty_func = "modulo8", convolution_type = "mixed", penalty_weight = 1, lr = 1e-3, n_epochs = 1000, compute_if_not_persisted = False):
    """
    Segmentation results on the RWC Pop when the latent space and the sparsity parameter are fitted by cross validation, on ``learning dataset''.
    Results are shown for the test dataset.
    """
    if learning_dataset == "odd_songs":
        learning_dataset_path = paths.path_odd_songs_rwc
    elif learning_dataset == "even_songs":
        learning_dataset_path = paths.path_even_songs_rwc
    elif learning_dataset == "debug":
        learning_dataset_path = paths.path_debug_rwc
    else:
        raise err.InvalidArgumentValueException(f"Dataset type not understood: {learning_dataset}") from None
        
    if testing_dataset == "odd_songs":
        testing_dataset_path = paths.path_odd_songs_rwc
    elif testing_dataset == "even_songs":
        testing_dataset_path = paths.path_even_songs_rwc
    elif testing_dataset == "debug":
        testing_dataset_path = paths.path_debug_rwc
    else:
        raise err.InvalidArgumentValueException(f"Dataset type not understood: {testing_dataset_path}") from None
        
    if learning_dataset == testing_dataset:
        warnings.warn("Careful: using the same dataset as test and learn, normal?")
        
    annotations_type = "MIREX10"
    annotations_folder = "{}/{}".format(paths.path_annotation_rwc, annotations_type)
    hop_length = 32
    hop_length_seconds = 32/44100
    
    learning_dataset_song_list = scr.load_RWC_dataset(learning_dataset_path, annotations_type)

    f_mes_zero_five = math.inf * np.ones((len(learning_dataset_song_list), len(dim_latent_space_range), len(sparsity_range), 1))
    
    for idx_song, song_and_annotations in enumerate(learning_dataset_song_list):
        #printmd('**Chanson courante: {}**'.format(song_and_annotations[0]))
        song_number = song_and_annotations[0].replace(".wav","")
        
        annot_path = "{}/{}".format(annotations_folder, song_and_annotations[1])
        annotations = dm.get_segmentation_from_txt(annot_path, annotations_type)
        references_segments = np.array(annotations)[:, 0:2]
        
        song_path = f"{learning_dataset}/{song_number}.wav"
        
        spectrogram = scr.load_or_save_spectrogram(paths.path_data_persisted_rwc, song_path, feature, hop_length)    
        bars = scr.load_or_save_bars(paths.path_data_persisted_rwc, song_path)
                
        tensor_spectrogram = nn_utils.tensorize_barwise(spectrogram, bars, hop_length_seconds, subdivision)
        data_loader = nn_utils.generate_dataloader(tensor_spectrogram)
        
        for idx_dls, dim_latent_space in enumerate(dim_latent_space_range):
            for idx_lambda_s, sparsity_lambda in enumerate(sparsity_range):
                
                projection_conv = scr.load_or_save_convolutional_projection(paths.path_data_persisted_rwc, song_number, data_loader, dim_latent_space, sparsity_lambda = sparsity_lambda, nn = False, lr = lr, n_epochs = n_epochs, feature = feature, hop_length = hop_length, subdivision_bars = subdivision, compute_if_not_persisted = compute_if_not_persisted)
                projection_conv = np.where(abs(projection_conv) < 1e-15, 0, projection_conv)
                autosimilarity = as_seg.get_autosimilarity(projection_conv, transpose = True, normalize = True)
                
                segments = as_seg.dynamic_convolution_computation(autosimilarity, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
                segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                
                prec, rap, f_mes = dm.compute_score_of_segmentation(references_segments, segments_in_time, window_length = 0.5)
                f_mes_zero_five[idx_song, idx_dls, idx_lambda_s] = round(f_mes,4)
                    
    best_sparses_arg = []
    best_fmes_sparse = []
    
    for idx_dls, dim_latent_space in enumerate(dim_latent_space_range):
        fmes_sparse = []
        for idx_lambda_s, sparsity_lambda in enumerate(sparsity_range):
            mean_sparse = np.mean(f_mes_zero_five[:,idx_dls,idx_lambda_s,0])
            fmes_sparse.append(mean_sparse)
        best_fmes_arg = np.argmax(fmes_sparse)
        best_sparses_arg.append(best_fmes_arg)
        best_fmes_sparse.append(fmes_sparse[best_fmes_arg])
    best_dim_latent_space_arg = np.argmax(best_fmes_sparse)
    best_dim_latent_space = dim_latent_space_range[best_dim_latent_space_arg]
    best_sparse_best_dls = sparsity_range[best_sparses_arg[best_dim_latent_space_arg]]
    
    display(pd.DataFrame(np.array([best_dim_latent_space, best_sparse_best_dls]), index = ['Best latent space dimension','Best sparsity value'], columns = ["Learned parameters"]).T)

    results_at_zero_five, results_at_three = final_results_fixed_conditions_rwc(testing_dataset, feature, best_dim_latent_space, best_sparse_best_dls, lr = lr, n_epochs = n_epochs, subdivision = 96, penalty_func = penalty_func, penalty_weight = penalty_weight, convolution_type = convolution_type, compute_if_not_persisted = compute_if_not_persisted, legend = " on test dataset.")
    
    return results_at_zero_five, results_at_three


# %% Tests on SALAMI dataset
def fixed_conditions_results_salami(feature = "log_mel_grill", sparsity_lambda = None, lr = 1e-3, dim_latent_space = 16, penalty_weight = 1, convolution_type = "mixed", penalty_func = "modulo8", compute_if_not_persisted = False):
    """
    Computing segmentation results with AE projection on the SALAMI test subset.
    """            
    salami = mirdata.initialize('salami', data_home = paths.path_entire_salami)
    len_salami = len(salami.track_ids)

    res_ae = -math.inf * np.ones((len_salami, 2, 3))
    res_baselines = -math.inf * np.ones((len_salami, 2, 3))
    
    hop_length = 32
    hop_length_seconds = hop_length/44100
    subdivision_bars = 96
    n_epochs = 1000
    
    all_tracks = salami.load_tracks()
    
    song_idx = 0
    
    file_mirex = open(f"{os.getcwd()}/list_salami_mirex.txt")

    test_dataset = []
    for part in file_mirex.readlines():
        line_broken = part.split("\n")
        test_dataset.append(int(line_broken[0]))
            
    for key, track in all_tracks.items():
        if int(key) in test_dataset:
            try:               
                bars = scr.load_or_save_bars(paths.path_data_persisted_salami, track.audio_path)
                tensor_barwise = scr.load_or_save_tensor_spectrogram(paths.path_data_persisted_salami, track.audio_path, feature, hop_length, subdivision_bars)
                data_loader = nn_utils.generate_dataloader(tensor_barwise)

                ref_tab = []
                try:
                    references_segments = salami.load_sections(track.sections_annotator1_uppercase_path).intervals
                    ref_tab.append(references_segments)
                except (TypeError, AttributeError):
                    pass
                    
                try:
                    references_segments = salami.load_sections(track.sections_annotator2_uppercase_path).intervals
                    ref_tab.append(references_segments)
                except (TypeError, AttributeError):
                    pass
                
                # Baseline signal
                autosimilarity_signal = as_seg.get_autosimilarity(tl.unfold(tensor_barwise,0), transpose = True, normalize = True)
                segments = as_seg.dynamic_convolution_computation(autosimilarity_signal, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
                segments_in_time = dm.segments_from_bar_to_time(segments, bars)
                    
                score_baseline_zero_five = dm.compute_score_of_segmentation(ref_tab[0], segments_in_time, window_length = 0.5)
                score_baseline_three = dm.compute_score_of_segmentation(ref_tab[0], segments_in_time, window_length = 3)
                if len(ref_tab) > 1:
                    second_score_three = dm.compute_score_of_segmentation(ref_tab[1], segments_in_time, window_length = 3)
                    if second_score_three[2] > score_baseline_three[2]: # f measure
                        score_baseline_zero_five = dm.compute_score_of_segmentation(ref_tab[1], segments_in_time, window_length = 0.5)
                        score_baseline_three = second_score_three 
                
                res_baselines[song_idx, 0] = score_baseline_zero_five 
                res_baselines[song_idx, 1] = score_baseline_three
                    
                # AE score
                projection = scr.load_or_save_convolutional_projection(paths.path_data_persisted_salami, key, data_loader, dim_latent_space, sparsity_lambda = sparsity_lambda, nn = False, lr = lr, n_epochs = n_epochs, feature = feature, hop_length = hop_length, subdivision_bars = subdivision_bars, compute_if_not_persisted = compute_if_not_persisted)
                autosimilarity_ae = as_seg.get_autosimilarity(projection, transpose = True, normalize = True)
                segments_ae = as_seg.dynamic_convolution_computation(autosimilarity_ae, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
                segments_ae_in_time = dm.segments_from_bar_to_time(segments_ae, bars)
                
                score_ae_zero_five = dm.compute_score_of_segmentation(ref_tab[0], segments_ae_in_time, window_length = 0.5)
                score_ae_three = dm.compute_score_of_segmentation(ref_tab[0], segments_ae_in_time, window_length = 3)
                
                if len(ref_tab) > 1:
                    second_score_ae_three = dm.compute_score_of_segmentation(ref_tab[1], segments_ae_in_time, window_length = 3)
                    if second_score_ae_three[2] > score_ae_three[2]: # 3sec f measure
                        score_ae_zero_five = dm.compute_score_of_segmentation(ref_tab[1], segments_ae_in_time, window_length = 0.5)
                        score_ae_three = second_score_ae_three

                res_ae[song_idx, 0] = score_ae_zero_five
                res_ae[song_idx, 1] = score_ae_three
           
                song_idx += 1  
    
            except FileNotFoundError:
                print(f"{key} not found, normal ?")

    print(f"Tested on {song_idx} songs")
    lines = np.array(["Precision 0.5", "Recall 0.5", "F measure 0.5","Precision 3", "Recall 3", "F measure 3"])  
    tab = []
    bas_np = res_baselines[:song_idx]
    tab.append([round(np.mean(bas_np[:,0, 0]),5), round(np.mean(bas_np[:, 0,1]),5), round(np.mean(bas_np[:, 0,2]),5), round(np.mean(bas_np[:,1, 0]),5), round(np.mean(bas_np[:, 1,1]),5), round(np.mean(bas_np[:, 1,2]),5)])
    all_res_np = res_ae[:song_idx]
    tab.append([round(np.mean(all_res_np[:,0, 0]),5), round(np.mean(all_res_np[:, 0,1]),5), round(np.mean(all_res_np[:, 0,2]),5), round(np.mean(all_res_np[:,1, 0]),5), round(np.mean(all_res_np[:, 1,1]),5), round(np.mean(all_res_np[:, 1,2]),5)])

    display(pd.DataFrame(tab, index=["On barwise feature", "On the latent representation"], columns=lines))

        
def learn_and_test_salami(feature = "log_mel_grill", sparsity_lambda = None, lr = 1e-3, list_dim_latent_space = [16], penalty_weight = 1, convolution_type = "mixed", penalty_func = "modulo8", compute_if_not_persisted = False):
    """
    Learning the latent space dimension on the non-test subset of the SALAMI dataset, and computes the results on the SALAMI test-dataset.
    """       
    if sparsity_lambda != None:
        raise NotImplementedError("Case where sparsity is implemented has not been developed for SALAMI yet (won't learn this parameter), so TODO") from None
    
    salami = mirdata.initialize('salami', data_home = paths.path_entire_salami)
    len_salami = len(salami.track_ids)
    
    res_ae = -math.inf * np.ones((len_salami, len(list_dim_latent_space), 2, 3))
    res_baselines = -math.inf * np.ones((len_salami, 2, 3))
    
    hop_length = 32
    hop_length_seconds = hop_length/44100
    subdivision_bars = 96
    n_epochs = 1000
    
    all_tracks = salami.load_tracks()    
    song_idx = 0
    file_mirex = open(f"{os.getcwd()}/list_salami_mirex.txt")

    test_dataset = []
    for part in file_mirex.readlines():
        line_broken = part.split("\n")
        test_dataset.append(int(line_broken[0]))
            
    for key, track in all_tracks.items():
        if int(key) not in test_dataset: # Every other file than the test dataset
            try:
                bars = scr.load_or_save_bars(paths.path_data_persisted_salami, track.audio_path)
                tensor_barwise = scr.load_or_save_tensor_spectrogram(paths.path_data_persisted_salami, track.audio_path, feature, hop_length, subdivision_bars)
                data_loader = nn_utils.generate_dataloader(tensor_barwise)

                ref_tab = []
                try:
                    references_segments = salami.load_sections(track.sections_annotator1_uppercase_path).intervals
                    ref_tab.append(references_segments)
                except (TypeError, AttributeError):
                    pass
                    
                try:
                    references_segments = salami.load_sections(track.sections_annotator2_uppercase_path).intervals
                    ref_tab.append(references_segments)
                except (TypeError, AttributeError):
                    pass
                
                for idx_dls, dim_latent_space in enumerate(list_dim_latent_space):
                    projection = scr.load_or_save_convolutional_projection(paths.path_data_persisted_salami, key, data_loader, dim_latent_space, sparsity_lambda = sparsity_lambda, nn = False, lr = lr, n_epochs = n_epochs, feature = feature, hop_length = hop_length, subdivision_bars = subdivision_bars, compute_if_not_persisted = compute_if_not_persisted)
                    autosimilarity_ae = as_seg.get_autosimilarity(projection, transpose = True, normalize = True)
                                    
                    segments_ae = as_seg.dynamic_convolution_computation(autosimilarity_ae, penalty_weight = penalty_weight, penalty_func = penalty_func, convolution_type = convolution_type)[0]                
                    segments_ae_in_time = dm.segments_from_bar_to_time(segments_ae, bars)
                    
                    score_ae_zero_five = dm.compute_score_of_segmentation(ref_tab[0], segments_ae_in_time, window_length = 0.5)
                    score_ae_three = dm.compute_score_of_segmentation(ref_tab[0], segments_ae_in_time, window_length = 3)
                    
                    if len(ref_tab) > 1:
                        second_score_ae_three = dm.compute_score_of_segmentation(ref_tab[1], segments_ae_in_time, window_length = 3)
                        if second_score_ae_three[2] > score_ae_three[2]: # 3sec f measure
                            score_ae_zero_five = dm.compute_score_of_segmentation(ref_tab[1], segments_ae_in_time, window_length = 0.5)
                            score_ae_three = second_score_ae_three
    
                    res_ae[song_idx, idx_dls, 0] = score_ae_zero_five
                    res_ae[song_idx, idx_dls, 1] = score_ae_three
               
                song_idx += 1  
    
            except FileNotFoundError:
                print(f"{key} not found, normal ?")

    mean_dls = []
    for idx_dls, dim_latent_space in enumerate(list_dim_latent_space):
        mean_dls.append(round(np.mean(res_ae[:song_idx, idx_dls, 1, 2]), 5)) # Best at 3sec tolerance
    best_dim_latent_space_arg = np.argmax(mean_dls)
    best_dim_latent_space = list_dim_latent_space[best_dim_latent_space_arg]
    
    display(pd.DataFrame(np.array([best_dim_latent_space, song_idx]), index = ['Best latent space dimension', 'Number of songs in learning dataset'], columns = ["Learned parameters"]).T)

    return fixed_conditions_results_salami(feature = feature, sparsity_lambda = sparsity_lambda, lr = lr, dim_latent_space = best_dim_latent_space, penalty_weight = penalty_weight, convolution_type = convolution_type, penalty_func = penalty_func)    
