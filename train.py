import numpy as np
import tqdm as tqdm
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torch.utils.tensorboard import SummaryWriter

import config

from model import *
from utils import *
from STL import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _train(my_model, dataset, dataloader, filename):

        writer = SummaryWriter('runs/' + filename)
        my_model.train()

        if config.GAN:
            my_discriminator = Discriminator2d()
            my_discriminator.train()

        #I set the initial learning rate
        lr = config.LR_MAX

        if config.GRADIENT:
            gradient_list = []
            for j in range(1 + 4*4 + 1):
                gradient_list.append([])

        for i in tqdm(range(config.EPOCHS)):

            with open("control_epoch.txt", "w") as output:
                output.write(str(i))

            my_model.epochs = i
            sum_loss, sum_loss_grad = 0, 0

            #If we use the progressive training we start with the cosine annealing only after 120 epochs
            if config.PROGRESSIVE and i >= 120:
                lr = config.LR_MIN + 0.5*(config.LR_MAX - config.LR_MIN)*(1 + np.cos((i - 120)*np.pi/(config.EPOCHS - 120)))

            #Otherwise we start the cosine annealing immediately
            if config.PROGRESSIVE == False:
                lr = config.LR_MIN + 0.5*(config.LR_MAX - config.LR_MIN)*(1 + np.cos(i*np.pi/config.EPOCHS))

            for param_group in my_model.optimizer.param_groups:
                param_group['lr'] = lr

            if config.GAN:
                for param_group in my_discriminator.optimizer.param_groups:
                    param_group['lr'] = lr

            a = 0
            counter = 0
            
            if config.GRADIENT:
                my_gradients = [0 for j in range(1 + 4*4 + 1)]
            
            for (input, label) in dataloader:
                counter += 1

                p = min(config.NUM_BIG_BLOCKS - 1, int(my_model.epochs/30))
                
                my_model.optimizer.zero_grad()
                if config.GAN: my_discriminator.optimizer.zero_grad()

                label = label.to(my_model.device)

                #We need to downsample the label velocity fields appropriately
                if config.PROGRESSIVE and p < config.NUM_BIG_BLOCKS - 1:
                    label = torchvision.transforms.Resize(size=(16 * 2**p, 12 * 2**p)).forward(label)

                if config.PROGRESSIVE and p > 0 and int(my_model.epochs/30) < config.NUM_BIG_BLOCKS and (my_model.epochs % 30) < 20:
                    
                    pred_0, pred_1 = my_model(input)
                    pred_0 = torchvision.transforms.Resize(size=(pred_1.shape[-2], pred_1.shape[-1])).forward(label)

                    a = (i*dataset.__len__() + counter)/(20.0*dataset.__len__()) - 1.5 * p
                    pred = (1-a)*pred_0 + a*pred_1 #I have the smooth transition

                else:
                    pred = my_model(input)

                if config.GAN:

                    result_discr_fake = my_discriminator(torch.cat((pred, vorticity(pred)), dim = 1))
                    loss_gen = config.GAN_WEIGHT * torch.mean(torch.square(result_discr_fake - torch.ones(result_discr_fake.shape).to(my_model.device))) + \
                                        torch.mean(torch.abs(pred - label)) + \
                                        torch.mean(torch.abs(jacobian2d(pred) - jacobian2d(label)))

                    loss_gen.backward()
                    my_model.optimizer.step()

                    my_discriminator.optimizer.zero_grad()

                    fake_input = torch.cat((pred, vorticity(pred)), dim = 1).detach()
                    label_input = torch.cat((label, vorticity(label)), dim = 1)

                    result_discr_fake = my_discriminator(fake_input)
                    result_discr_gt = my_discriminator(label_input)
                    
                    loss_discr = torch.mean(torch.square(result_discr_fake)) + \
                                 torch.mean(torch.square(result_discr_gt - torch.ones(result_discr_gt.shape).to(my_model.device)))

                    loss_discr.backward()
                    my_discriminator.optimizer.step()
                    continue

                if config.STL:
                    loss = config.STL_WEIGHTS[0]*get_frq_loss(my_model.bands[0], label, pred)
                    for i in range(1, len(my_model.bands)):
                        loss += config.STL_WEIGHTS[i]*get_frq_loss(my_model.bands[i], label, pred)

                else:
                    
                    loss = torch.mean(torch.abs(pred - label))
                    loss += torch.mean(torch.abs(jacobian2d(pred) - jacobian2d(label)))

                    sum_loss += torch.mean(torch.abs(pred - label)).item()
                    sum_loss_grad += torch.mean(torch.abs(jacobian2d(pred) - jacobian2d(label))).item()

                loss.backward()

                if config.GRADIENT:

                    my_gradients[0] += torch.norm(my_model.Linear_Layer.weight.grad, p=1).item() \
                                        /count_parameters(my_model.Linear_Layer)

                    for j in range(config.NUM_BIG_BLOCKS):
                        for k in range(config.CONV_LAYERS_PER_BB):
                            my_gradients[4*j + k + 1] += torch.norm(my_model.big_blocks[j].conv_layers[k].weight.grad, p=1).item()\
                                                        /count_parameters(my_model.big_blocks[j].conv_layers[k])

                    my_gradients[-1] += torch.norm(my_model.to_RGB.weight.grad).item()\
                                        /count_parameters(my_model.to_RGB)

                my_model.optimizer.step()

            writer.add_scalar('loss', sum_loss/dataset.__len__() * config.BATCH_SIZE, i)
            writer.add_scalar('grad_loss', sum_loss_grad/dataset.__len__() * config.BATCH_SIZE, i)

            if config.GRADIENT:

                writer.add_scalar('Gradient Linear Layer', my_gradients[0]/dataset.__len__() * config.BATCH_SIZE, i)
                gradient_list[0].append(my_gradients[0]/dataset.__len__() * config.BATCH_SIZE)

                for j in range(config.NUM_BIG_BLOCKS):
                        for k in range(config.CONV_LAYERS_PER_BB):
                            writer.add_scalar('Big Block nbr: ' + str(j) + ' Conv_layer nbr: ' + str(k) , my_gradients[4*j + k + 1]/dataset.__len__() * config.BATCH_SIZE, i)
                            gradient_list[4*j + k + 1].append(my_gradients[4*j + k + 1]/dataset.__len__() * config.BATCH_SIZE)

                writer.add_scalar('Gradient RGB_layer', my_gradients[-1]/dataset.__len__() * config.BATCH_SIZE, i)
                gradient_list[-1].append(my_gradients[-1]/dataset.__len__() * config.BATCH_SIZE)

        if config.GRADIENT:
        
            for i in range(len(gradient_list)):
                #print(i)
                if i == 0:
                    np.savetxt('my_gradients/Linear_Layer', gradient_list[0])

                elif i == len(gradient_list) - 1:
                    np.savetxt('my_gradients/To_RGB', gradient_list[-1])

                else:
                    np.savetxt('my_gradients/Big_Block_' + str(int((i-1)/config.NUM_BIG_BLOCKS) + 1) + '_Conv_Layer_' + str((i-1)%4), gradient_list[i])

