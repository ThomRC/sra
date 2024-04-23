# SRA
 Code from "Smoothed Robustness Analysis" paper

# Basic requirements

- CUDA enabled GPU with at least 3GB of RAM to reproduce the experiments from the paper
- Conda to easily create the environment with all required packages
- Git to clone the repository

# Conda setup
Follow instructions from https://docs.conda.io/projects/miniconda/en/latest/:

    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh

Run the following to initialize conda

    ~/miniconda3/bin/conda init bash

Run the following to create an environment named "sra-env" with all the required packages

    conda create -n sra-env -c conda-forge python=3.8 chainer=7.8.0 cupy=7.8.0 scikit-learn=0.23 scipy=1.7

Run the following to activate the environment in order to run the provided code

    conda activate sra-env

# Clone github repository
In the terminal change your directory to the one you want to clone the repository

    git clone https://github.com/ThomRC/sra.git

Go to the repository so you can run the experiments.py file

    cd sra

# How to use
The only python file that you need to run is experiments.py

This file requires five positional arguments like:
    
    python experiments.py [loss] [d] [x_var] [mode] [gpu]

### loss 

    'mscr', 'mcr', 'zhen', 'mh', 'sce', 'scsce'

### d 

    hyperparameter used in margin based losses to control the maximal margin that contributes to the loss

### x_var

    hyperparameter used in smoothed losses to control the smoothing effect of adding an isotropic Gaussian noise to the input

### mode

    'train' carries the training of NNs with the specified loss and hyperparameters

    'load' loads the trained NNs with the specified loss and hyperparameters to be then subject to the robustness measurements

    'load_all' loads all trained NNs to be then subject to the robustness measurements

#### Note 1: 
The first time you run the training it might take some minutes for the first epoch to finish. The cause is that cupy compiles all the CUDA functions to be used when the NN is generated

#### Note 2: 
The architecture of the NN for training is the same used in the paper, in case you want to modify it you do it in the exp_settings.py file 

#### Note 3: 
When using 'load' or 'load_all', before the measurements start you will be prompted with a confirmation of dataset to be used and architecture. This is used in case you want to test different datasets or architectures to avoid mistakes when carrying the measurements  

### gpu
    
    Number of the GPU to be used for the training. When in load mode the used gpu is the same from the training even if the gpu argument is another one


# Code References
denselinear.py, bjorck_linear.py, selections.py, bjorck_ortho.py based on github/cemanil, respectively - 
https://github.com/cemanil/LNets/blob/master/lnets/models/layers/dense/base_dense_linear.py
https://github.com/cemanil/LNets/blob/master/lnets/models/layers/dense/bjorck_linear.py
https://github.com/cemanil/LNets/blob/master/lnets/models/utils/selections.py
https://github.com/cemanil/LNets/blob/master/lnets/utils/math/projections/l2_ball.py
