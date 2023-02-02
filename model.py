import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

import config
from utils import *
from STL import *

def weights_init(m):
    torch.nn.init.xavier_uniform_(m.weight)
    torch.nn.init.zeros_(m.bias)

class BigBlock(nn.Module):

    def __init__(self):

        super(BigBlock, self).__init__()
        
        self.conv_layers = nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(config.CONV_LAYERS_PER_BB):
            self.conv_layers.append(nn.Conv2d(config.NUM_CHANNELS, config.NUM_CHANNELS, 3, padding=1).apply(weights_init))

    def forward(self, x):

        for i in range(config.CONV_LAYERS_PER_BB):
            x = self.leaky(self.conv_layers[i](x))

        return x

class Generator2d(nn.Module):
    
    def __init__(self, STL = False, GAN = False, PROGRESSIVE = False, sum = -1):
    
        super(Generator2d, self).__init__()

        self.STL, self.GAN, self.PROGRESSIVE = STL, GAN, PROGRESSIVE

        self.epoch = 0 #The number of elapsed epochs, it will be used for the progressive training

        if self.STL: 
            self.bands = get_rect_bands()

        self.pre_reshape_dim = 128 * 8 * 6 # (n_channels, height, width) of the first feature map
        self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        
        if config.PE == True:
            self.Linear_Layer = nn.Linear(3 + 6 * sum, self.pre_reshape_dim).apply(weights_init)

        else:
            self.Linear_Layer = nn.Linear(3, self.pre_reshape_dim).apply(weights_init)
        
        self.big_blocks = nn.ModuleList([])

        for i in range(config.NUM_BIG_BLOCKS):
            self.big_blocks.append(BigBlock())

        if self.PROGRESSIVE:
            self.ds_RGB_list = nn.ModuleList([])
            for i in range(len(self.big_blocks)):
                 self.ds_RGB_list.append(nn.Conv2d(config.NUM_CHANNELS, 1, 3, padding=1).apply(weights_init))

        else:
            self.to_RGB = nn.Conv2d(config.NUM_CHANNELS, 1, 3, padding=1).apply(weights_init)

        self.optimizer = optim.Adam(self.parameters(), betas=(config.BETA_1, config.BETA_2))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):

        if torch.cuda.is_available():
            x = torch.tensor(x).to(self.device) #GPU
        else:
            x = torch.tensor(x) #CPU

        #Project the input array to a first feature map
        out = self.Linear_Layer(x)
        out = torch.reshape(out, (-1, 128, 8, 6))
        out_0 = out # This will be used for the skip connection

        if self.PROGRESSIVE and config.TRAIN == True:
            p = min(config.NUM_BIG_BLOCKS - 1, int(self.epochs/30))

        else:
            p = config.NUM_BIG_BLOCKS - 1

        for i in range(p + 1):
            
            #In the progressive training there are no skip connections
            if self.PROGRESSIVE:
                out = self.big_blocks[i].forward(out)
            else:
                out = self.big_blocks[i].forward(out) + out_0

            #We only upsample for the right amount of times, until the desired resolution is attained
            if i < math.log2(config.Y_RES / 8):
                out = self.upsampler(out)
            
            if i != p:
                out_0 = out
        
        if self.PROGRESSIVE and config.TRAIN == True:
            if p > 0 and int(self.epochs/30) < len(self.big_blocks) and (self.epochs % 30) < 20:
                return curl2d(self.ds_RGB_list[p-1](out_0)), curl2d(self.ds_RGB_list[p](out))
            else:
                return curl2d(self.ds_RGB_list[p](out))

        elif self.PROGRESSIVE and config.TRAIN == False:
            return curl2d(self.ds_RGB_list[p](out))

        else:
            return curl2d(self.to_RGB(out))

class Discriminator2d(nn.Module):
    def __init__(self):
        super(Discriminator2d, self).__init__()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.conv_layers = nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        channels = [3, 64, 128, 256, 512, 1]
        strides = [2, 2, 2, 1, 1]

        #Mi da errore sul 3 di channels, perchè vuole anche la vorticity!
        #Che io al momento non ho implementato

        for i in range(len(channels) - 1):
            self.conv_layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1, stride=strides[i]).apply(weights_init))

        self.optimizer = optim.Adam(self.parameters(), betas=(config.BETA_1, config.BETA_2))
        self.to(self.device)

    def forward(self, x):

        for i in range(len(self.conv_layers)):
            if i != len(self.conv_layers) - 1:
                x = self.leaky(self.conv_layers[i](x))
            else:
                x = self.conv_layers[i](x)

        return x

if __name__ == "__main__":
    my_gen = Generator2d()