## Overview

This project ports the existing Deep Fluids architecture: https://arxiv.org/abs/1806.02071 from Tensorflow to PyTorch, and focuses on improving the reconstruction of fine grain details of fluids' velocity fields. 

Major visual improvements are attained thanks to the use of Positional Encoding and Progressive Training: https://arxiv.org/abs/1710.10196. The employed technique largely outperforms, both quantitatively and qualitatively, previous approaches which aimed at enhancing the same architecture: https://deepai.org/publication/frequency-aware-reconstruction-of-fluid-simulations-with-generative-networks

## Architecture

The architecture's input consists of a vector *c* containing just three numbers:

-The position of the fluid source on the x axis (the y position is kept fixed across the whole dataset)

-The size of the source of fluid

-The time elapsed since the source was activated

Up-scalings and convolutions are performed on the input vector *c* to reconstruct the whole velocity field of the fluid. Finally the curl operator is applied to ensure that the final output field is divergence free. The project focuses on reconstructing 2D velocity fields, but can be easily extended to higher dimensions. THe fields are visualized as images, with the x and y components of the velocity being associated respectively with colors magenta and green. The full CNN architecture structure is presented below.

<p align="center">
    <img src=./images/CNN.png width="600" />
</p>

## Qualitative Improvements

In the 1st column Ground Truth (GT) fields are presented.

In the 2nd column Baseline reconstructions, obtained via the vanilla implementation of Deep Fluids, are shown.

In the 3rd column STL (Shift Towards Low) reconstrictions are visible. These are the reconstructions obtained via a technique previously developed to improve Deep Fluids' ability to reconstruct fine grain details. This approach is described at: https://deepai.org/publication/frequency-aware-reconstruction-of-fluid-simulations-with-generative-networks

In the 4th and 5th column my results are shown. In these cases the *c* vector was preprocessed via Positional Encoding (PE) combined with STL (fourth column) or Progressive Training (fifth column).

It is clearly visible that Positional Encoding is a game changer. The reconstructed fields of the 4th and 5th columns present much more details than both Baseline and STL, with the best results being those attained via the combined use of Positional Encoding and Progressive Training.

<p align="center">
<img src=./images/Mixed_Fields.png width="600" />
</p>

## Quantitative Improvements

The curves in the graph on the left are associated with different architectures. Each curve represents the relative difference between the Discrete Fast Fourier Transform of the reconstructed fields and the ground truth one, averaged across the whole dataset, at different frequencies. As it is visible, Positional Encoding combined with Progressive Training is the approach which reduces the curve's area the most, bringing a reduction in covered area that is 4 times as large as that obtained with the previous appraoch (STL).

On the right are displayed five histograms reporting the counts of different values for the log magnitude in frequency space. It can be observed that the statistics of the frequencies, for the dataset reconstructed via Positional Encoding + Progressive Training, is closer to the Ground Truth statistics as compared to previous approaches.

<p align="center">
<img src=./images/stats.PNG width="600" />
</p>

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
