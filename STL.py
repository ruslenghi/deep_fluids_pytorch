import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import cv2 as cv
import shutil

def get_rect_bands():

    bands = []
    c_h, c_w = int(128/2), int(96/2) #Center of height and center of width

    #Because we have 5 bands
    for i in range(5):
        
        i_th_band = np.zeros((1, 2, 128, 96))
        i_th_band[:, :, c_h-4*2**i:c_h+4*2**i, c_w-3*2**i:c_w+3*2**i] = np.ones((1, 2, 4*2**(i+1), 3*2**(i+1)))
        for j in range(len(bands)):
            i_th_band -= bands[j]
        bands.append(i_th_band)

    return bands

def circular_band_radius(r):
    a = np.zeros((1, 2, 128, 96))
    for i in range(128):
        for j in range(96):
            if ((i-63.5)**2 + (j-47.5)**2) < r*r:
                a[0][0][i][j] = 1
                a[0][1][i][j] = 1
    
    return a

def filter(band, image):

    image = torch.fft.fft2(image)
    image = torch.fft.fftshift(image, (2,3))
    image = image * band
    #image = torch.fft.ifftshift(image, (2,3))
    #image = torch.fft.ifft2(image)

    return image

def get_MRE(band, gt, reconstructed):
    
    band = torch.tensor(band)

    if torch.cuda.is_available():
        batch_band = torch.cat(gt.shape[0]*[band]).to(torch.device('cuda:0'))
    else:
        batch_band = torch.cat(gt.shape[0]*[band])
    
    c_gt = filter(batch_band, gt)
    c_reconstructed = filter(batch_band, reconstructed)

    MRE = torch.sum(torch.abs(c_gt-c_reconstructed)).item()/torch.sum(torch.abs(c_gt)).item()

    return MRE

def get_frq_loss(band, gt, reconstructed):
    
    band = torch.tensor(band)
    if torch.cuda.is_available():
        batch_band = torch.cat(gt.shape[0]*[band]).to(torch.device('cuda:0'))
    else:
        batch_band = torch.cat(gt.shape[0]*[band])
    
    c_reconstructed = filter(batch_band, reconstructed)
    c_gt = filter(batch_band, gt)

    my_loss = torch.sum(torch.abs(c_gt - c_reconstructed))
    
    return my_loss