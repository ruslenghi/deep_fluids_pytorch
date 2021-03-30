from PIL import Image
import torch
import numpy as np
from model import *

def print_images(my_model, smoke_images, physical_info, max_v):
    
    np.random.seed(1)
    my_model.eval()

    for k in range(20):

        r_n = np.random.randint(len(smoke_images))
        v_ = torch.Tensor(physical_info[r_n])

        my_result = my_model.forward(v_)

        my_result = my_result.cpu()
        x_r = my_result.detach().numpy()
        x_r = np.reshape(x_r, [2, 128, 96])
        x_r = x_r.transpose(1, 2, 0)
        x_r = np.concatenate([x_r, np.zeros([128, 96, 1])], axis=-1)
        x_r = np.clip((x_r + 1)*127.5, 0, 255)
        x_r = x_r[::-1]

        im = Image.fromarray(x_r.astype(np.uint8))
        im.save('result/reconstructed_' + str(r_n) + '_.png')


        gt = smoke_images[r_n].transpose(1,2,0)
        gt = np.concatenate([gt, np.zeros([128, 96, 1])], axis=-1)
        gt = np.clip((gt + 1)*127.5, 0, 255)
        gt = gt[::-1]

        im = Image.fromarray(gt.astype(np.uint8))
        im.save('result/gt_' + str(r_n) + '_.png')