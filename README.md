## Overview

This project ports the existing Deep Fluids architecture: https://github.com/byungsook/deep-fluids from Tensorflow to PyTorch, and focuses on improving the reconstruction of fine grain details of fluids' velocity fields. 

Major visual improvements are attained thanks to the use of Positional Encoding and Progressive Training: https://arxiv.org/abs/1710.10196. The employed technique largely outperforms, both quantitatively and qualitatively, previous approaches which aimed at enhancing the same architecture: https://deepai.org/publication/frequency-aware-reconstruction-of-fluid-simulations-with-generative-networks

## Architecture

The architecture's input consists of a vector *c* containing just three numbers:

-The position of the fluid source on the x axis (the y position is kept fixed across the whole dataset)

-The size of the source of fluid

-The time elapsed since the source was activated

Up-scalings and convolutions are performed on the input vector *c* to reconstruct the whole velocity field of the fluid. Finally the curl operator is applied to ensure that the final output field is divergence free. 

The project focuses on reconstructing 2D velocity fields, but can be easily extended to higher dimensions. The fields are visualized as images, with the x and y components of the velocity being associated respectively with colors magenta and green.

The CNN architecture's structure is shown below.

<p align="center">
    <img src=./images/CNN.png width="600" />
</p>

## Qualitative Improvements

In the 1st column Ground Truth (GT) fields are presented.

In the 2nd column Baseline reconstructions, obtained via the vanilla implementation of Deep Fluids, are shown.

In the 3rd column STL (Shift Towards Low) reconstrictions are visible. These are the reconstructions obtained via a technique previously developed to improve Deep Fluids' ability to reconstruct fine grain details. This approach is described at: https://deepai.org/publication/frequency-aware-reconstruction-of-fluid-simulations-with-generative-networks

In the 4th and 5th column my results are shown. In these cases the *c* vector was preprocessed via Positional Encoding (PE) and the network was trained with STL (fourth column) or Progressive Training (fifth column).

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

This code is tested on Windows 10 and Mac OS.

Run the following line to install required packages.

    pip install --upgrade torch torchvision tqdm matplotlib opencv-python tensorboard

## Usage

Run the following line to train the model

    python main.py my_model
    
Run the following line to reconstruct random dataset samples using the trained model.
    
    python print_fields.py my_model
    
After this, randomly selected reconstructions of dataset samples will be located in the "results" folder.
    
## Dataset

In this repo a toy dataset called "super_reduced", which contains only 10 velocity fields, was loaded. Training the model for 240 epochs on this toy dataset using my Macbook Air 2021 takes around 3 minutes, and yields trustworthy reconstructions.

The images and graphs shown in this README were however obtained by training the model on a dataset of 21000 velocity fields. 

To train the model on larger datasets follow the prescriptions presented at: https://github.com/byungsook/deep-fluids.
Then, save the obtained dataset in a sub-folder called "data", put it in the same folder as the load_data.py script, open the config.py file and change the DATASET value to "data". Then run the commands described at the Usage stage.
