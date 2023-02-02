import torch
import sys
import glob
import shutil
import tqdm as tqdm
import matplotlib.pyplot as plt

from load_data import *
from model import *
from STL import *
from utils import *


for np_name in glob.glob('my_gradients/*'):
    a = np.loadtxt(np_name)
    x = [i for i in range(len(a))]
    plt.plot(x, a, c='r')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient')
    plt.title(np_name[13:])
    plt.savefig('my_gradients/' + np_name[13:] + '_gradients.png', dpi=300)
    plt.clf()


