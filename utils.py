import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import config

def error_message():

    if config.GAN and config.PROGRESSIVE:
        print('The joint use of GAN and Progressive training is not supported. \n')
        print('Please open the config.py file and set config.GAN or config.PROGRESSIVE to False')
        exit()

    if config.GAN and config.NUM_BIG_BLOCKS != 5:
        print('The GAN is not generalized to be run with an arbitrary number of big blocks. \n')
        print('Please open the config.py file and set config.NUM_BIG_BLOCKS = 5')
        exit()

    '''if config.GAN == False and config.NUM_BIG_BLOCKS == 5:
        print('You are using 5 Bigblocks without the GAN, you sure? \n')
        print('Please open the config.py file and set config.NUM_BIG_BLOCKS = 4')
        exit()'''
    
    if config.STL and config.PROGRESSIVE:
        print('The joint use of STL and Progressive training is not supported. \n')
        print('Please open the config.py file and set config.STL or config.PROGRESSIVE to False')
        exit()
    

def curl2d(x):

    u = x[:, :, 1:, :] - x[: , :, :-1, :] # -ds/dy,
    v = x[: , :, :, :-1] - x[: , :, :, 1:] # ds/dx
    u = torch.cat([u, torch.unsqueeze(u[:, : , -1, :], axis=2)], axis=2)
    v = torch.cat([v, torch.unsqueeze(v[:, :, :, -1], axis=3)], axis=3)
    c = torch.cat([u,v], axis=1)

    return c


def jacobian2d(x):

    dudy = x[:, 0, 1:, :] - x[:, 0, :-1, :]
    dudy = torch.cat((dudy, torch.unsqueeze(dudy[:, -1, :], 1)), dim=1)
    
    dudx = x[:, 0, :, 1:] - x[:, 0, :, :-1]
    dudx = torch.cat((dudx, torch.unsqueeze(dudx[:, :, -1], 2)), dim=2)

    dvdy = x[:, 1, 1:, :] - x[:, 1, :-1, :]
    dvdy = torch.cat((dvdy, torch.unsqueeze(dvdy[:, -1, :], 1)), dim=1)
    
    dvdx = x[:, 1, :, 1:] - x[:, 1, :, :-1]
    dvdx = torch.cat((dvdx, torch.unsqueeze(dvdx[:, :, -1], 2)), dim=2)

    j = torch.cat((dudx, dudy, dvdx, dvdy))

    return j

def vorticity(x):

    dudy = x[:, 0, 1:, :] - x[:, 0, :-1, :]
    dudy = torch.cat((dudy, torch.unsqueeze(dudy[:, -1, :], 1)), dim=1)
    
    dvdx = x[:, 1, :, 1:] - x[:, 1, :, :-1]
    dvdx = torch.cat((dvdx, torch.unsqueeze(dvdx[:, :, -1], 2)), dim=2)

    w = dvdx - dudy
    w = torch.unsqueeze(w, dim=1)

    return w

def turn_into_RGB(x, fs):

    h, w = config.Y_RES, config.X_RES

    x = x.transpose(1, 2, 0)
    x = np.concatenate([x, np.zeros([h, w, 1])], axis=-1)
    x = np.clip((x + 1)*127.5, 0, 255)
    x = x[::-1]
    framed_x = np.zeros([h + 2*fs, w + 2*fs, 3])
    framed_x[fs:h+fs, fs:w+fs, :] = x

    return framed_x

def print_images(my_model, smoke_images, physical_info):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    fs = 1 #Frame size

    alphabet = [str(i) for i in range(len(smoke_images))]
    alphabet = sorted(alphabet)

    for i in tqdm(range(len(smoke_images))):
        
        v_ = torch.Tensor(physical_info[i]).to(device)
        
        x_r = my_model.forward(v_).cpu().detach().numpy()
        x_r = np.reshape(x_r, [2, x_r.shape[-2], x_r.shape[-1]])

        framed_x_r = turn_into_RGB(x_r, fs).repeat(4, axis=0).repeat(4, axis=1)
        #framed_gt = turn_into_RGB(smoke_images[i], fs)

        #merged = np.concatenate((framed_x_r, framed_gt), axis=1).repeat(4, axis=0).repeat(4, axis=1)
        merged = framed_x_r

        im = Image.fromarray(merged.astype(np.uint8))
        im.save('result/RL_GTR' + alphabet[i] + '_.png')