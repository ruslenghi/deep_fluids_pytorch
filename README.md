## Overview

This project ports the existing Deep Fluids architecture: https://arxiv.org/abs/1806.02071 from Tensorflow to PyTorch, and focuses on improving the reconstruction of fine grain details of fluids' velocity fields. 

Major visual improvements are attained thanks to the use of Positional Encoding and Progressive Training. The final results largely outperform, both quantitatively and qualitatively, previous approaches which aimed at enhancing the same architecture: https://deepai.org/publication/frequency-aware-reconstruction-of-fluid-simulations-with-generative-networks

## Architecture

The architecture's input consists of a vector *c* containing just three numbers:

-The position of the fluid source on the x axis (the y position is kept fixed across the whole dataset)

-The size of the source of fluid

-The time elapsed since the source was activated

Up-scalings and convolutions are performed on the input vector *c* to reconstruct the whole velocity field of the fluid. 
The full CNN architecture structure is presented below.

<img src=./images/CNN.png width="600" />

## Qualitative Improvements

<img src=./images/Mixed_Fields.png width="600" />

## Quantitative Improvements

<img src=./images/stats.PNG width="600" />

## Requirements

This code is tested on Windows 10 with the following requirements:

<!-- - [anaconda / python3.6](https://www.anaconda.com/download/) (run `conda install python=3.6` for the latest version.) -->
- [TensorFlow 1.15](https://www.tensorflow.org/install/)
- [mantaflow](http://mantaflow.com)

Run the following line to install packages.

    pip install --upgrade tensorflow==1.15 tqdm matplotlib Pillow imageio

## Usage

    python main.py my_model
    
    python print_fields.py my_model
