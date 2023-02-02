import torch
import sys
import shutil
import tqdm as tqdm
import matplotlib.pyplot as plt

from load_data import *
from model import *
from STL import *
from utils import *

filename = sys.argv[1]

config.TRAIN = False

sum = 0
for i in config.POWER_LIST:
    sum += i

physical_info, smoke_images = load_and_preprocess(config.POWER_LIST)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Generator2d(STL=config.STL, GAN=config.GAN, PROGRESSIVE=config.PROGRESSIVE, sum = sum)
model = torch.load('my_models/' + filename + '.pt', map_location = device)
model.eval()

np.random.seed(1)

mra = np.zeros(78)

rings = []
for i in range(2, 80): 
    rings.append(circular_band_radius(i) - circular_band_radius(i-1))
rings = torch.tensor(rings).to(device)

counter = [i for i in range(len(rings))]

for i in tqdm(range(len(smoke_images))):

    v_ = torch.Tensor(physical_info[i])
    gt = torch.tensor(smoke_images[i]).to(device)
    pred = model.forward(v_)

    for j in range(len(rings)):
        mra[j] += get_MRE(rings[j], torch.reshape(gt, (1,2,128,96)), pred)/len(smoke_images)

np.savetxt(filename + '_mra.txt', mra)
plt.plot(counter, mra, c='b', label='Baseline')
plt.legend()
plt.savefig('Frequency_plot.png')
plt.show()

'''mra_old = np.loadtxt('old_guess_mra.txt')
mra_best = np.loadtxt('best_guess_mra.txt')

plt.plot(counter, mra_old, c='b', label='Old')
plt.plot(counter, mra_best, c='r', label='Best')
plt.legend()
plt.savefig('Frequency_plot.png')
plt.show()'''
