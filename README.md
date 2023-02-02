## Description

In the present work we focus on enhancing the reconstruction of high frequency detail for velocity fields obtained via generative data driven fluid simulations. In particular we employ Progressive Training and Positional Encoding as methods to improve detail reconstruction. Both these techniques largely outperform the benchmark method we consider and can be combined to provide even better results. The observed quantitative improvement in frequency reconstruction is effectively reflected by a much larger amount of visual detail in the generated velocity fields.

<img src=./images/CNN.png width="600" />

test test test

<img src=./images/Mixed_Fields.png width="600" />

test test test

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
