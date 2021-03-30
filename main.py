#from PIL import Image
import torch
#import torch.nn as nn 
#import torch.nn.functional as F
#import torch.optim as optim
from torchvision.transforms import ToTensor
import numpy as np
#import matplotlib.pyplot as plt
import sys
#import os
import shutil
from torch.utils.data import TensorDataset, DataLoader, Dataset

import sys
from load_data import *
from model import *
from trainer import _train #For some reason 'from trainer import *' doesn't work
from get_results import print_images

#PROGRAM USAGE: python main.py 'dataset you want to use' 'number of epochs' 

velocity_dataset = sys.argv[1]
n_epochs = int(sys.argv[2])
physical_info, smoke_images, max_v = load_and_preprocess(velocity_dataset)

#make the arrays of interest become torch tensors
tensor_phys = torch.Tensor(physical_info) # transform to torch tensor
tensor_smoke = torch.Tensor(smoke_images)

#prepare the dataset and dataloader
my_dataset = TensorDataset(tensor_phys, tensor_smoke) # create your datset
my_dataloader = DataLoader(my_dataset, batch_size = 8, shuffle=True) # create your dataloader

#I create a folder to store some results of reconstruction of randomly
#selected images from the training set
dir_ = 'result'
if os.path.exists(dir_):
    shutil.rmtree(dir_)
os.makedirs(dir_)
print('Hello, me here!')

my_model = CNN(lr=0.0001, epochs=n_epochs)
_train(my_model, my_dataset, my_dataloader)
print_images(my_model, smoke_images, physical_info, max_v)