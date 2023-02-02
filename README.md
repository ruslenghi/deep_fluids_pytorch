The reduced.py file is a script that produces 2000 smoke images (different among them for source size and position).
The 'result_2000_images_400_epochs_8_batchsize' contains reconstructions and gt of images from the 2000 smoke images dataset.
The main.py can be run to reproduce the said results (almost, I have not fixed any random seed, so the reconstructed images will be different).

## Requirements

This code is tested on Windows 10 with the following requirements:

<!-- - [anaconda / python3.6](https://www.anaconda.com/download/) (run `conda install python=3.6` for the latest version.) -->
- [TensorFlow 1.15](https://www.tensorflow.org/install/)
- [mantaflow](http://mantaflow.com)

Run the following line to install packages.

    $ pip install --upgrade tensorflow==1.15 tqdm matplotlib Pillow imageio

To install `mantaflow`, run:

    $ git clone https://bitbucket.org/mantaflow/manta.git
    $ git checkout 15eaf4
    
and follow the [instruction](http://mantaflow.com/install.html). Note that `numpy` cmake option should be set to enable support for numpy arrays. (i.e., `-DNUMPY='ON'`)
