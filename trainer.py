import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from model import *


def _train(my_model, dataset, dataloader):
        my_model.train()#This tells pytorch that we are in training mode

        losses = []
        grad = []
        x = []
        for i in range(my_model.epochs):
            print('This is i: ', i)
            c = 0
            avg_grad = 0
            x.append(i)
            for (input, label) in dataloader:
                
                my_model.optimizer.zero_grad()
                label = label.to(my_model.device)
                prediction = my_model.forward(input)
                loss = my_model.loss(prediction, label)
                c += loss.item()
                loss.backward()

                g = 0
                counter = 0
                for p in my_model.parameters():
                    g += p.grad.norm()
                    counter += 1
                avg_grad += g
                my_model.optimizer.step()
            
            losses.append(c/dataset.__len__())
            grad.append(avg_grad/dataset.__len__())

        plt.plot(x, losses)
        plt.savefig('result/loss.png')
        plt.clf()

        plt.plot(x, grad)
        plt.savefig('result/grad.png')
        plt.clf()