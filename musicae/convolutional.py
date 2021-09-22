# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:46:42 2021

@author: amarmore
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import musicae.neural_nets_utils as nn_utils
from musicae.model.early_stopping import EarlyStopping
import musicae.model.current_plot as current_plot
import musicae.model.errors as err

import random
import warnings

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class ConvolutionalAutoencoder(nn.Module):
    """
    Class defining the Convolutional Neural Network (CNN) used to compress barwise representation of a song.
    
    This CNN is developed uisng the PyTorch framework.
    """
    def __init__(self, input_size_x, input_size_y, dim_latent_space = 16, nonnegative = False):
        """
        Constructor of the network.

        Parameters
        ----------
        input_size_x : int
            The x-axis size of the input matrix.
        input_size_y : int
            The y-axis size of the input matrix.
        dim_latent_space : int, optional
            Dimension of the latent space. The default is 16.
        nonnegative : boolean, optional
            DEPRECATED. Indicating whether to construct nonnegative latent vectors. The default is False.

        Raises
        ------
        err.OutdatedBehaviorException.
            Raised if ``nonnegative'' argument is set to True.

        Returns
        -------
        None.

        """
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.input_size_pool_y = int(input_size_y/4) # input_size / pool ## NOTE: Doesn't work for odd input sizes (in reconstruction), so TODO.
        self.input_size_pool_x = int(input_size_x/4) # input_size / pool
        
        # Encoder
        ## Convolutional layers
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)  
        self.conv2 = nn.Conv2d(4,16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        ## Fully-connected controling latent space size.
        in_features = 16 * self.input_size_pool_x * self.input_size_pool_y # nb_kernels_last_conv * (input_size / pool)
        self.fc = nn.Linear(in_features=in_features, out_features=dim_latent_space)
        
        # Decoder
        ## Inverse of previous FC
        self.i_fc = nn.Linear(in_features=dim_latent_space, out_features=in_features)

        # Transposed conv layers
        self.t_conv1 = nn.ConvTranspose2d(16, 4, 3, stride=2, padding=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1)
        
        if nonnegative:
            raise err.OutdatedBehaviorException("So-called ``nonnegative networks'' are not supported anymore, as they weren't succesful.") from None
        
        # self.nonnegative = nonnegative

    def forward(self, x):
        """
        Forward pass.
        x_hat is the reconstructed input, z is the latent projection.
        """
        batch_size = x.shape[0]
        
        # Encoding
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.flatten(x)
        
        ## Nonnegative is an old beavior.
        # if self.nonnegative:
        #     z = F.softplus(self.fc(x))
        # else:
        z = self.fc(x)
            
        # Decoding
        x = self.i_fc(z)
                
        x = x.view((batch_size, 16, self.input_size_pool_y, self.input_size_pool_x))
        x = F.relu(self.t_conv1(x))
        x_hat = self.t_conv2(x)
        return x_hat, z
    
    def my_optim_method(self, n_epochs, data_loader, lr = 1e-3, early_stop_patience = 100, verbose = False, labels = None):
        """
        Optimization method. Defined directly in the class, in order to be called from the object.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to perform.
        data_loader : torch.DataLoader
            The DataLoader to optimize on.
        lr : float, optional
            Learning rate of the Network. The default is 1e-3.
        early_stop_patience : int, optional
            Patience for the number of consecutive epochs.
            If the loss doesn't decrease during early_stop_patience epochs, the optimization stops. The default is 100.
        verbose : boolean, optional
            Argument to print the evolution of the optimization.
            Prints the current loss and plots a view of the autosimilarity of latent variables and a PCA of the latent space.
            The default is False.
        labels : None or array, optional
            Only used if verbose is set to True.
            If labels are set, they will be used to color the output of PCA projection.
            If they are set to None, no label is used. The default is None.

        Returns
        -------
        ConvolutionalAutoencoder
            Returns the network.
            Note: not mandatory (as the object needs to be instanciated to call this method), but returned as a good practice.

        """
        es = EarlyStopping(patience=early_stop_patience)

        recons_loss = nn.MSELoss() # Mean Squared Euclidian loss
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Scheduler to decrease the learning rate when optimization reaches a plateau (20 iterations)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=verbose,min_lr=1e-5) 
        self.apply(nn_utils.seeded_weights_init_kaiming) # Seeded initialization of the network
        for epoch in range(1, n_epochs+1):
            total_loss_epoch = 0.0
            for spec in data_loader:
                spec = spec.float()
                optimizer.zero_grad()
                spec_recons, z = self.forward(spec) # Forward pass
                loss = recons_loss(spec_recons, spec) # Loss
                loss.backward() # Backward pass
                optimizer.step() # Optimizes the net with grads
                
                total_loss_epoch += loss.item() #*spec.size(0)
            scheduler.step(total_loss_epoch) # scheduler for the decrease of lr on a plateau
                
            total_loss_epoch = total_loss_epoch/len(data_loader)
            if verbose:
                print('Epoch: {} \tCumulated reconstruction loss: {:.6f}'.format(epoch, total_loss_epoch))
                if epoch%50 == 0:
                    projection = self.get_latent_projection(data_loader)
                    current_plot.plot_latent_space(projection, labels = labels)
                    
            if es.step(total_loss_epoch): # Checks if loss has decreased, to stop the optimization if performances don't increase for early_stop_patience epochs.
                if verbose:
                    print(f"Early stopping criterion has been met in {epoch}, computation is stopped.")
                break 
        return self
    
    def get_latent_projection(self, data_loader):
        """
        Returns the latent projection of elements in data_loader.

        Parameters
        ----------
        data_loader : torch.DataLoader
            The objects to project in latent space.

        Returns
        -------
        all_data : numpy.array
            Projection of the data_loader, as a numpy array.
        """
        all_data = []
        for spec in data_loader:
            spec = spec.float()
            spec_recons, z = self.forward(spec)
            for elt in z:
                all_data.append(elt.detach().numpy())
        return all_data
    
class ConvolutionalAutoencoderSparse(ConvolutionalAutoencoder):
    """
    Class defining the Sparse Convolutional Neural Network (SAE) used to compress barwise representation of a song.
    
    This SAE is the same CNN than the previously defined (``ConvolutionalAutoencoder'') but adds a saprsity constraint on the latent representation.
    The goal of this sparsity is to disentangle the different dimensions of the latent space.
    """
    def __init__(self, input_size_x, input_size_y, dim_latent_space, nonnegative, sparsity_lambda, norm_tensor):
        """
        Constructor of the network.

        Parameters
        ----------
        input_size_x : int
            The x-axis size of the input matrix.
        input_size_y : int
            The y-axis size of the input matrix.
        dim_latent_space : int, optional
            Dimension of the latent space. The default is 16.
        nonnegative : boolean, optional
            DEPRECATED. Indicating whether to construct nonnegative latent vectors. The default is False.
        sparsity_lambda : float
            Ponderation parameter for the sparsity constraint (on z).
        norm_tensor : float
            Norm of the tensor on which to optimize, used as a normalization for the sparsity parameter.

        Raises
        ------
        err.OutdatedBehaviorException.
            Raised if ``nonnegative'' argument is set to True.

        Returns
        -------
        None.

        """
        if nonnegative:
            raise err.OutdatedBehaviorException("So-called ``nonnegative networks'' are not supported anymore, as they weren't succesful.") from None
        
        super().__init__(input_size_x, input_size_y, dim_latent_space, nonnegative = False)#, nonnegative)
        self.sparsity_lambda_normed = sparsity_lambda #round(sparsity_lambda/norm_tensor, 8) ## Normalized sparsity parameter

    def my_optim_method(self, n_epochs, data_loader, lr=1e-2, verbose = False, labels = None, early_stop_patience = 100):
        """
        Optimization method. Defined directly in the class, in order to be called from the object.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to perform.
        data_loader : torch.DataLoader
            The DataLoader to optimize on.
        lr : float, optional
            Learning rate of the Network. The default is 1e-3.
        early_stop_patience : int, optional
            Patience for the number of consecutive epochs.
            If the loss doesn't improve during early_stop_patience epochs, the optimization stops. The default is 100.
        verbose : boolean, optional
            Argument to print the evolution of the optimization.
            Prints the current loss and plot a view of the autosimilarity of latent variables and a PCA of the latent space.
            The default is False.
        labels : None or array, optional
            Only used if verbose is set to True.
            If labels are set, they will be used to color the output of PCA projection.
            If they are set to None, no label is used. The default is None.

        Returns
        -------
        ConvolutionalAutoencoderSparse
            Returns the Network.
            Note: not mandatory (as the object needs to be instanciated to call this method), but returned as a good practice.

        """
        es = EarlyStopping(patience=early_stop_patience)
        
        recons_loss = nn.MSELoss() # Mean Squared Euclidian loss
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Scheduler to decrease the learning rate when optimization reaches a plateau (20 iterations)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=verbose,min_lr=1e-5) 
        self.apply(nn_utils.seeded_weights_init_kaiming) # Seeded initialization of the network

        for epoch in range(1, n_epochs+1):
            total_loss_epoch = 0.0
            for spec in data_loader:
                spec = spec.float()
        
                optimizer.zero_grad()
                x_hat, z = self.forward(spec) # Forward
                recons_loss_spec = recons_loss(x_hat, spec) # Reconstruction loss
                l1_regularization = torch.abs(z).mean() # L1 reg
                loss = recons_loss_spec + self.sparsity_lambda_normed * l1_regularization # Mixed loss
                loss.backward() # Backward
                optimizer.step()
                total_loss_epoch += loss.item()
            scheduler.step(total_loss_epoch)
            
            if verbose:
                total_loss_epoch = total_loss_epoch/len(data_loader)
                print('Epoch: {} \tCumulated reconstruction loss: {:.6f}'.format(epoch, total_loss_epoch))
                if epoch%50 == 0:
                    projection = self.get_latent_projection(data_loader)
                    current_plot.plot_latent_space(projection, labels = labels)
                    
            if es.step(total_loss_epoch):
                if verbose:
                    print(f"Early stopping criterion has been met in {epoch}, computation is stopped.")
                break 
        return self
    
#### DEPRECATED: Triplet loss autoencoders (Weren't successful).
class ConvolutionalAutoencoderTriplet(ConvolutionalAutoencoder):
    """
    Class defining AutoEncoders (AE) used to compress barwise representation of a song, with an added Triplet loss to regularize the latent space.
    This triplet loss was designed to 
    """
    def __init__(self, input_size_x, input_size_y, dim_latent_space, nonnegative, triplet_lambda = 1, triplet_type = "simple", triplet_margins = [1]):
        """
        Constructor of the network.

        Parameters
        ----------
        input_size_x : int
            The x-axis size of the input matrix.
        input_size_y : int
            The y-axis size of the input matrix.
        dim_latent_space : int, optional
            Dimension of the latent space. The default is 16.
        nonnegative : boolean, optional
            DEPRECATED. Indicating whether to construct nonnegative latent vectors. The default is False.
        triplet_lambda : float, optional
            Ponderation for the triplet loss in the mixed loss. The default is 1.
        triplet_type : string, optional
            Type of triplet loss, either simple or double.
             - "simple" refer to the standard triplet loss,
             - "double" refer to the triplet loss with a positive and a negative margin.
             Both triplet losses are introduced in [1], respectively as triplet loss 1 and triplet loss 3.
            The default is "simple".
        triplet_margins : array of ints, optional
            Margins for both triplet losses.
            "simple" triplet loss uses one argument in the array, and "double" uses two, in the order [positive_margin, negative_margin].
            The default is [1].

        Raises
        ------
        err.OutdatedBehaviorException.
            Raised if ``nonnegative'' argument is set to True.
        err.InvalidArgumentValueException.
            Raised if the triplet loss type is not understood.

        Returns
        -------
        None.
        
        References
        ----------
        [1] Ho, K., Keuper, J., Pfreundt, F. J., & Keuper, M. (2021, January). 
        Learning embeddings for image clustering: An empirical study of triplet loss approaches. 
        In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 87-94). IEEE.

        """
        if nonnegative:
            raise err.OutdatedBehaviorException("So-called ``nonnegative networks'' are not supported anymore, as they weren't succesful.") from None
        
        super().__init__(input_size_x, input_size_y, dim_latent_space, nonnegative)
        self.triplet_lambda = triplet_lambda
        if triplet_type == "simple":
            self.triplet_function = nn_utils.TripletLoss(margin = triplet_margins[0])
        elif triplet_type == "double":
            self.triplet_function = nn_utils.TripletLossDoubleMargin(pos_margin = triplet_margins[0], neg_margin = triplet_margins[1])
        else:
            raise err.InvalidArgumentValueException(f"Triplet loss type not understood: {triplet_type}")

    def forward(self, anchor, positive, negative):
        """
        Forward pass.
        Succession of forward passes for each input (anchor, positive and negative).
        Returns all reconstructed input and latent projections.
        """
        x_hat_a, z_a = super().forward(anchor)
        x_hat_p, z_p = super().forward(positive)
        x_hat_n, z_n = super().forward(negative)
        return x_hat_a, z_a, x_hat_p, z_p, x_hat_n, z_n
    
    def my_optim_method(self, n_epochs, data_loader, lr=1e-2, regenerate_data_loader = True, tensor_barwise = None, verbose = False, labels = None, early_stop_patience = 100):
        """
        Optimization method. Defined directly in the class, in order to be called from the object.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to perform.
        data_loader : torch.DataLoader
            The DataLoader to optimize on.
        lr : float, optional
            Learning rate of the Network. The default is 1e-3.
        regenerate_data_loader : boolean, optional
            Boolean specifying whether the positive and negative examples needs to be recomputed at each iteration.
            The default is True.
        tensor_barwise : None or numpy.array, optional
            Only used if regenerate_data_loader is set to True.
            Original tensor_barwise, on which is compute dthe optimization, to regenrate positive and negative examples. 
            The default is None.
        early_stop_patience : int, optional
            Patience for the number of consecutive epochs.
            If the loss doesn't improve during early_stop_patience epochs, the optimization stops. The default is 100.
        verbose : boolean, optional
            Argument to print the evolution of the optimization.
            Prints the current loss and plot a view of the autosimilarity of latent variables and a PCA of the latent space.
            The default is False.
        labels : boolean, optional
            Only used if verbose is set to True.
            If labels are set, they will be used to color the output of PCA projection.
            The default is None.

        Returns
        -------
        ConvolutionalAutoencoder
            Returns the Network.
            Note: not mandatory (as the object needs to be instanciated to call this method), but returned as a good practice.

        """
        es = EarlyStopping(patience=early_stop_patience)
        recons_loss = nn.MSELoss()

        triplet_loss_function = torch.jit.script(self.triplet_function)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=verbose,min_lr=1e-5)
        self.apply(nn_utils.seeded_weights_init_kaiming)

        for epoch in range(1, n_epochs+1):
            total_loss_epoch = 0.0
            if regenerate_data_loader:
                data_loader, _ = nn_utils.generate_triplet_dataloader(tensor_barwise)
            for anchor, positive, negative in data_loader:
                anchor = torch.reshape(anchor, (anchor.shape[0],1, anchor.shape[1],anchor.shape[2]))
                anchor = anchor.float()
                positive = torch.reshape(positive, (positive.shape[0],1, positive.shape[1],positive.shape[2]))
                positive = positive.float()
                negative = torch.reshape(negative, (negative.shape[0],1, negative.shape[1],negative.shape[2]))
                negative = negative.float()
                
                optimizer.zero_grad()
                
                x_hat_a, z_a, x_hat_p, z_p, x_hat_n, z_n = self.forward(anchor, positive, negative) # Forward
        
                # Recon losses
                recons_loss_a = recons_loss(x_hat_a, anchor)
                recons_loss_p = recons_loss(x_hat_p, positive)
                recons_loss_n = recons_loss(x_hat_n, negative)
                # Triplet loss
                triplet_loss_val = triplet_loss_function(z_a, z_p, z_n)
                # Mixed loss
                loss = recons_loss_a + recons_loss_p + recons_loss_n + self.triplet_lambda * triplet_loss_val
                loss.backward() # Backward
                optimizer.step()
                total_loss_epoch += loss.item()
            scheduler.step(total_loss_epoch)
            
            if verbose:
                total_loss_epoch = total_loss_epoch/len(data_loader)
                print('Epoch: {} \tCumulated reconstruction loss: {:.6f}'.format(epoch,total_loss_epoch))
                
                if epoch%50 == 0:
                    projection = self.get_latent_projection(data_loader)
                    current_plot.plot_latent_space(projection, labels = labels)
                    
            if es.step(total_loss_epoch):
                if verbose:
                    print(f"Early stopping criterion has been met in {epoch}, computation is stopped.")
                break 
        return self
    
    def get_latent_projection(self, data_loader):
        """
        Returns the latent projection of elements in data_loader.

        Parameters
        ----------
        data_loader : torch.DataLoader
            The objects to project in latent space.

        Returns
        -------
        all_data : numpy.array
            Projection of the data_loader, as a numpy array.

        """
        all_data = []
        for spec, _, _ in data_loader:
            spec = spec.float()
            spec = torch.reshape(spec, (spec.shape[0], 1, spec.shape[1], spec.shape[2]))
            spec_recons, z = super().forward(spec)
            for elt in z:
                all_data.append(elt.detach().numpy())
        return all_data

    
# class ConvolutionalAutoencoderTripletSparse(ConvolutionalAutoencoderTriplet):
#     def __init__(self, input_size_x, input_size_y, dim_latent_space, nonnegative, triplet_lambda = 1, triplet_type = "simple", triplet_margins = [1], sparsity_lambda = 1):
#         super().__init__(input_size_x, input_size_y, dim_latent_space, nonnegative, triplet_lambda = triplet_lambda, triplet_type = triplet_type, triplet_margins = triplet_margins)
#         self.sparsity_lambda = sparsity_lambda
    
#     def my_optim_method(self, n_epochs, data_loader, lr=1e-2, regenerate_data_loader = True, tensor_barwise = None, verbose = False, labels = None, early_stop_patience = 100):
#         es = EarlyStopping(patience=early_stop_patience)
#         recons_loss = nn.MSELoss()
#         triplet_loss_function = torch.jit.script(self.triplet_function)
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=verbose,min_lr=1e-5)
#         self.apply(nn_utils.seeded_weights_init_kaiming)
#         # elif init == "xavier":
#         #     self.apply(nn_utils.seeded_weights_init_xavier)
#         # else:
#         #     raise NotImplementedError("Init not understood")
        
#         for epoch in range(1, n_epochs+1):
#             total_loss_epoch = 0.0
#             if regenerate_data_loader:
#                 data_loader, _ = nn_utils.generate_triplet_dataloader(tensor_barwise)
#             for anchor, positive, negative in data_loader:
#                 anchor = torch.reshape(anchor, (anchor.shape[0],1, anchor.shape[1],anchor.shape[2]))
#                 anchor = anchor.float()
#                 positive = torch.reshape(positive, (positive.shape[0],1, positive.shape[1],positive.shape[2]))
#                 positive = positive.float()
#                 negative = torch.reshape(negative, (negative.shape[0],1, negative.shape[1],negative.shape[2]))
#                 negative = negative.float()
                
#                 optimizer.zero_grad() # Clear grads
#                 x_hat_a, z_a, x_hat_p, z_p, x_hat_n, z_n = self.forward(anchor, positive, negative) # Forward
        
#                 # Recon losses
#                 recons_loss_a = recons_loss(x_hat_a, anchor)
#                 recons_loss_p = recons_loss(x_hat_p, positive)
#                 recons_loss_n = recons_loss(x_hat_n, negative)
#                 # Triplet loss
#                 triplet_loss_val = triplet_loss_function(z_a, z_p, z_n)
#                 #L1 reg
#                 l1_regularization = torch.abs(z_a).mean() + torch.abs(z_p).mean() + torch.abs(z_n).mean()
#                 #Mixed loss
#                 loss = recons_loss_a + recons_loss_p + recons_loss_n + self.triplet_lambda * triplet_loss_val + self.sparsity_lambda * l1_regularization
#                 loss.backward() # Backward
#                 optimizer.step() # Optimization
#                 total_loss_epoch += loss.item() # Total loss
#             scheduler.step(total_loss_epoch)
            
#             if verbose:
#                 total_loss_epoch = total_loss_epoch/len(data_loader)
#                 print('Epoch: {} \tCumulated reconstruction loss: {:.6f}'.format(epoch,total_loss_epoch))
                
#                 if epoch%50 == 0:
#                     projection = self.get_latent_projection(data_loader)
#                     current_plot.plot_latent_space(projection, labels = labels)
                    
#             if es.step(total_loss_epoch):
#                 if verbose:
#                     print(f"Early stopping criterion has been met in {epoch}, computation is stopped.")
#                 break 
            
#         return self