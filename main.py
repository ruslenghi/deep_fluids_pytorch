import torch
from torchvision.transforms import ToTensor
import numpy as np
import sys
import shutil
from torch.utils.data import TensorDataset, DataLoader, Dataset

import sys
from load_data import *
from model import *
from train import _train
import config

if config.GRADIENT:
    dir_ = 'my_gradients'
    if os.path.exists(dir_): 
        pass
    else: 
        os.makedirs(dir_)

#The name with which you want to save the final model
filename = sys.argv[1]

#In case some of the configuration parameters are incompatible
#an error message is outputted and the program is ended
error_message()

sum = 0
for i in config.POWER_LIST:
    sum += i

physical_info, smoke_images = load_and_preprocess(config.POWER_LIST)

#make the arrays of interest become torch tensors
tensor_phys = torch.Tensor(physical_info) # transform to torch tensor
tensor_smoke = torch.Tensor(smoke_images)

#prepare the dataset and dataloader
my_dataset = TensorDataset(tensor_phys, tensor_smoke) # create your datset
my_dataloader = DataLoader(my_dataset, batch_size = config.BATCH_SIZE, shuffle=True) # create your dataloader

#Create and train the model
my_model = Generator2d(STL=config.STL, GAN=config.GAN, PROGRESSIVE=config.PROGRESSIVE, sum = sum)
_train(my_model, my_dataset, my_dataloader, filename)

dir_ = 'my_models'
if os.path.exists(dir_): 
    pass
else: 
    os.makedirs(dir_)

#Save the model
torch.save(my_model, 'my_models/' + filename + '.pt')