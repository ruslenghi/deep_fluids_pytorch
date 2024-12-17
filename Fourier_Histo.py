import torch
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm

from load_data import *
from model import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

filename = sys.argv[1]

physical_info, smoke_images = load_and_preprocess()

model = Generator2d()
my_model =  torch.load('my_models/' + filename + '.pt', map_location = device)
my_model.eval()

my_bins = 100

histo_reconstructed = np.zeros(my_bins)
histo_true = np.zeros(my_bins)
x_axis = np.zeros(my_bins)

np.random.seed(1)

for i in tqdm(range(100)):

    v_ = torch.Tensor(physical_info[i])

    my_result = my_model.forward(v_)

    my_result = my_result.cpu()
    x_r = my_result.detach().numpy()

    x_r = np.reshape(x_r, [2, 128, 96])
    gt = smoke_images[i]

    fft_x_r = np.abs(np.fft.fft2(x_r))
    fft_gt = np.abs(np.fft.fft2(gt))

    hist_x_r = np.histogram(np.log(fft_x_r), bins=my_bins, range=(-10,7), density=False)
    histo_reconstructed += hist_x_r[0]

    hist_gt = np.histogram(np.log(fft_gt), bins=my_bins, range=(-10,7), density=False)
    histo_true += hist_gt[0]
    x_axis = hist_x_r[1][1:]

plt.plot(x_axis, histo_reconstructed, c='y', label='Baseline')
plt.plot(x_axis, histo_true, c='r', label='Ground Truth')
plt.xlabel("Log magnitude")
plt.ylabel("Counts")
plt.title("Fourier Log Magnitudes")
plt.legend()
plt.savefig('Log_mag_plot.png')
plt.show()
