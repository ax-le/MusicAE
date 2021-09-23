# MusicAE: Encoding songs with Autoencoders to reveal structure #

Hello, and welcome on this repository!

This project aims at autoenconding all bars in a song, and studies the latent representations of every bar to infer its structure.
In that sense, every latent space is specialized for a song, and do not aim at generalizing between different songs.

This repository contains code for the Autoencoders, developed in PyTorch, and segmentation methods based on autosimilarity segmentation, as presented in [1].

You can download it using pip, by typing:

    pip install musicae

Or download the source files.

This is a first release, and may contain bug. Comments are welcomed!

## Software version ##

This code was developed with Python 3.8.5, and some external libraries detailed in dependencies.txt. They should be installed automatically if this project is downloaded using pip.

## Example Notebook ##

An example notebook is available in the folder "Notebooks", and presents the song 'Come Together' with different features.

## Credits ##

Code was created by Axel Marmoret (<axel.marmoret@irisa.fr>), and strongly supported by Jeremy E. Cohen (<jeremy.cohen@irisa.fr>).

The technique in itself was also developed by Frédéric Bimbot (<bimbot@irisa.fr>).

## References ##
[1] Marmoret, A., Cohen, J., Bertin, N., & Bimbot, F. (2020, October). Uncovering Audio Patterns in Music with Nonnegative Tucker Decomposition for Structural Segmentation. In ISMIR 2020-21st International Society for Music Information Retrieval.
