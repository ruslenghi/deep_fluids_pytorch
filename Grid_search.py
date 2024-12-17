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

dir_ = 'Grid_search'
if os.path.exists(dir_):
    pass
else: 
    os.makedirs(dir_)

def DecimalToBinary(num, my_list):
     
    if num >= 1:
        DecimalToBinary(num // 2, my_list)
    my_list.append(num % 2)

    return my_list

error_message()

mra = np.zeros(78)
rings = []
for i in range(2, 80): 
    rings.append(circular_band_radius(i) - circular_band_radius(i-1))
rings = torch.tensor(rings).to(config.DEVICE)

counter = [i for i in range(len(rings))]
randnums = np.random.randint(0, 200, 200)

for i in range(64 + 1):

    with open("control_powers.txt", "w") as output:
        output.write(str(i))

    my_list = []
    my_list = DecimalToBinary(i, my_list)
    my_list = my_list[1:]
    my_list.reverse()

    sum = 0
    for n in my_list:
        sum += n

    physical_info, smoke_images = load_and_preprocess(my_list)

    #make the arrays of interest become torch tensors
    tensor_phys = torch.Tensor(physical_info) # transform to torch tensor
    tensor_smoke = torch.Tensor(smoke_images)

    #prepare the dataset and dataloader
    my_dataset = TensorDataset(tensor_phys, tensor_smoke) # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size = config.BATCH_SIZE, shuffle=True) # create your dataloader

    #Create and train the model
    my_model = Generator2d(STL=config.STL, GAN=config.GAN, PROGRESSIVE=config.PROGRESSIVE, sum = sum)
    _train(my_model, my_dataset, my_dataloader)

    for j in randnums:

        v_ = torch.Tensor(physical_info[j])
        gt = torch.tensor(smoke_images[j]).to(config.DEVICE)
        pred = my_model.forward(v_)

        for k in range(len(rings)):
            mra[k] += get_MRE(rings[k], torch.reshape(gt, (1,2,128,96)), pred)/len(randnums)

    np.savetxt('Grid_search/' + str(i) + '_mra.txt', mra)
    mra = np.zeros(78)

dir_ = 'my_models'
if os.path.exists(dir_): 
    pass
else: 
    os.makedirs(dir_)
