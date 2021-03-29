#STRUCTURE OF THE CNN

#Take the 3-d vector containing physical info as input
#Project it to a big vector of dimension (8x6x128) via a Linear Layer
#Reshape the resulting vector as an (8, 6, 128) tensor
#Feed it to the the first Conv layer (make sure the dimension is fine, it might be that only 4-d tensors are accepted)
#Apply lrelu. Repeat this process 4 or 5 times.
#Apply the skip connection sum and upsample by 2.
#Repeat this procedure other 4 times, remembering not to upsample nor to skip connect the last time.
#This will yield a collection of 128 (128, 96) images. 
#Apply a last convolutional layer (with just 1 or 2 filters, depending on how many channels you have) to reconstruct the final image!
#FIRST TEST: Train the CNN a lot on just one image, just to see if it works

from PIL import Image
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import shutil
from torch.utils.data import TensorDataset, DataLoader, Dataset

from typing import List, Dict
from pathlib import Path
import pickle


#IDEAS

#Maybe the learning rate is not fine?

#QUESTIONS:

#How does Byungsoo obtain the gt images? The gt image is just one image
#so how can I obtain it from the .npz files? Maybe it is the case that I should
#just reconstruct a combined version of the v_x and v_y fields? Or use these as 
#2 channels of an rgb image?

#ADVICE:

#Implement the epochs saving system of the images
#Compare my code to Byungsoo's
#The checkboard effect mainly comes from the first linear layer

#TO TRY

#Maybe the skip connections are not properly used?
#Try running Byungsoo's code with lr=0.0001 fixed and see what happens!
#Print the loss from Byungsoo's code
#Change the learning rate
#Try to run the thing for 800 epochs with the adaptive learing rate
#See how the CNN deals with the full dataset
#Try using size 32 batches

#THINGS TO BE ADJUSTED:

#Understand how the gt images are plotter in Byungsoo's code

#Understand why the cross artifact appears and adjust it
#it looks like the CNN reconstructs 4 pieces of the image 
#separately and then it pastes them one beside the other

#Is it normal that I get the chessboard artifact at first?

#Solve teh "CHECKERBOARD PROBLEM"

#Print images correctly
#Adjust the learning rate
#Use the gradient loss
#Use the curl
#Implement Simon's loss fctn

#REMEMBER THAT THIS PART ONLY WORKS FOR THE 2000 IMAGES DATASET!

physical_info = []
smoke_images = []

for np_name in glob.glob('./data/reduced/v/*'):
    a = np.load(np_name)
    physical_info.append(a['y'])
    #print(a['x'].shape)
    #Originally we have (128, 96, 2)
    #a_x = np.reshape(a['x'], (2, 128, 96))
    a_x = np.array(a['x'].transpose(2, 0, 1), dtype=np.float32)
    #a_x = a['x'].transpose(2, 128, 96)
    #print(a_x.shape)
    smoke_images.append(a_x)

mult_factor = np.maximum(abs(np.max(smoke_images)), abs(np.min(smoke_images)))
smoke_images /= mult_factor

#I do the preprocessing for the physical param input
for i in range(len(smoke_images)):
    physical_info[i][0] = (physical_info[i][0] - 0.2)/0.6 * 2 - 1
    physical_info[i][1] = (physical_info[i][1] - 0.06)/0.04 * 2 - 1
    physical_info[i][2] = (physical_info[i][2] - 0)/100 * 2 - 1

#print(physical_info)
 
tensor_phys = torch.Tensor(physical_info) # transform to torch tensor
tensor_smoke = torch.Tensor(smoke_images)

my_dataset = TensorDataset(tensor_phys, tensor_smoke) # create your datset
dataloader = DataLoader(my_dataset, batch_size = 8, shuffle=True) # create your dataloader

class CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pre_reshape_dim = 128*8*6

        self.Linear_Layer = nn.Linear(3, self.pre_reshape_dim)
        
        #First big block
        self.conv1_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1_4 = nn.Conv2d(128, 128, 3, padding=1)

        #Second big block
        self.conv2_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_4 = nn.Conv2d(128, 128, 3, padding=1)

        #Third big block
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_4 = nn.Conv2d(128, 128, 3, padding=1)

        #Fourth big block
        self.conv4_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_4 = nn.Conv2d(128, 128, 3, padding=1)

        #Fifth big block
        self.conv5_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_4 = nn.Conv2d(128, 128, 3, padding=1)

        #Last conv layer
        #The 2 is there to obtain a 2 channels image
        #if you want a 1 channel image set it to 1
        self.conv_last = nn.Conv2d(128, 2, 3, padding=1)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr, betas=(0.5, 0.999))
        self.loss = nn.L1Loss()
        self.to(self.device)

    def forward(self, x):

        x = torch.tensor(x).to(self.device)
        x = self.Linear_Layer(x)
        x = torch.reshape(x, (-1, 128, 8, 6))

        x0 = x
        upsampler = nn.Upsample(scale_factor=2, mode='nearest')

        #First big block
        x = self.conv1_1(x)
        x = F.leaky_relu(x)
        x = self.conv1_2(x)
        x = F.leaky_relu(x)
        x = self.conv1_3(x)
        x = F.leaky_relu(x)
        x = self.conv1_4(x)
        x = F.leaky_relu(x) #ADDED
        x += x0 # That is the skip connection
        x = upsampler(x) #This is the by 2 upsampling step. (1, 1, 8, 6) --> (1, 1, 16, 12)
        x0 = x

        #Second big block
        x = self.conv2_1(x)
        x = F.leaky_relu(x)
        x = self.conv2_2(x)
        x = F.leaky_relu(x)
        x = self.conv2_3(x)
        x = F.leaky_relu(x)
        x = self.conv2_4(x)
        x = F.leaky_relu(x) #ADDED
        x += x0 # That is the skip connection
        x = upsampler(x) #This is the by 2 upsampling step. (1, 1, 16, 12) --> (1, 1, 32, 24)
        x0 = x

        #Third big block
        x = self.conv3_1(x)
        x = F.leaky_relu(x)
        x = self.conv3_2(x)
        x = F.leaky_relu(x)
        x = self.conv3_3(x)
        x = F.leaky_relu(x)
        x = self.conv3_4(x)
        x = F.leaky_relu(x) #ADDED
        x += x0 # That is the skip connection
        x = upsampler(x) #This is the by 2 upsampling step. (1, 1, 32, 24) --> (1, 1, 64, 48)
        x0 = x

        #Fourth big block
        x = self.conv4_1(x)
        x = F.leaky_relu(x)
        x = self.conv4_2(x)
        x = F.leaky_relu(x)
        x = self.conv4_3(x)
        x = F.leaky_relu(x)
        x = self.conv4_4(x)
        x = F.leaky_relu(x)  #ADDED
        x += x0 # That is the skip connection
        x = upsampler(x) #This is the by 2 upsampling step. (1, 1, 64, 48) --> (1, 1, 128, 96)
        x0 = x #ADDED

        #Fifth (and last) big block
        x = self.conv5_1(x)
        x = F.leaky_relu(x)
        x = self.conv5_2(x)
        x = F.leaky_relu(x)
        x = self.conv5_3(x)
        x = F.leaky_relu(x)
        x = self.conv5_4(x)
        x = F.leaky_relu(x) #ADDED
        x += x0 #ADDED
        #In this last big block we avoid upsampling (as the desired dimension is already attained)

        x = self.conv_last(x)
        return x
    
    def _train(self):
        self.train()#This tells pytorch that we are in training mode

        losses = []
        grad = []
        x = []
        for i in range(self.epochs):
            #self.lr = 0.0001/i 
            #print('This is i: ', i)
            c = 0
            avg_grad = 0
            x.append(i)
            for (input, label) in dataloader:
                
                self.optimizer.zero_grad()
                label = label.to(self.device)
                prediction = self.forward(input)
                loss = self.loss(prediction, label)
                c += loss.item()
                loss.backward()

                g = 0
                counter = 0
                for p in self.parameters():
                    g += p.grad.norm()
                    counter += 1
                avg_grad += g
                self.optimizer.step()
            
            losses.append(c/my_dataset.__len__())
            grad.append(avg_grad/my_dataset.__len__())

        plt.plot(x, losses)
        plt.savefig('result/loss.png')
        plt.clf()

        plt.plot(x, grad)
        plt.savefig('result/grad.png')
        plt.clf()

#I create a folder to store some results of reconstruction of randomly
#selected images from the training set
dir_ = 'result'
if os.path.exists(dir_):
    shutil.rmtree(dir_)
os.makedirs(dir_)

model = CNN(lr=0.0001, epochs=120, batch_size=8)
model._train()

for k in range(20):
    r_n = np.random.randint(len(smoke_images))
    v_ = torch.Tensor(physical_info[r_n])

    my_result = model.forward(v_)
    my_result = torch.reshape(my_result, (128, 96, 2))

    my_result = my_result.cpu()
    x_r = my_result.detach().numpy()
    x_r *= 10.362
    print(x_r.shape)
    x_r = np.reshape(x_r, (2, 128, 96))
    factor = 127.5/np.max(x_r)
    x_r = (x_r*factor).astype(np.uint8)

    image_test = np.zeros([128,96,3], dtype=np.uint8)

    for i in range(128):
        for j in range(96):
            image_test[i][j][0] = 127 + x_r[0][i][j]
            image_test[i][j][1] = 127 + x_r[1][i][j]
            image_test[i][j][2] = 127

    image_test = image_test[::-1]
    im = Image.fromarray(image_test)
    im.save('result/reconstructed_' + str(r_n) + '_.png')

    gt = smoke_images[r_n]
    factor = 127.5/np.max(gt)
    gt = (gt*factor).astype(np.uint8)

    for i in range(128):
        for j in range(96):
            image_test[127-i][j][0] = gt[0][i][j] + 127
            image_test[127-i][j][1] = gt[1][i][j] + 127
            image_test[127-i][j][2] = 127

    im = Image.fromarray(image_test)
    im.save('result/gt_' + str(r_n) + '_.png')
