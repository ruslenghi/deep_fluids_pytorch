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
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import shutil
from torch.utils.data import TensorDataset, DataLoader

#ADVICE:

#Try to train the cnn on a dataset containig 10/100 images (done 10 , to do 100)
#Implement the batch system (done, hope it's right)

#THINGS TO BE ADJUSTED:

#Adjust the learning rate
#Use the gradient loss
#Use the curl
#Implement Simon's loss fctn
#Find a better way to visualize the images

class CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size):
        super(CNN, self).__init__()#When you use inhertince in Python you have to call super
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.loss_history = [] #This has to be an array, and will be useful when we plot our data
        self.acc_history = []
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

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.loss = nn.L1Loss()
        self.to(self.device)
        self.get_data()
    
    def get_data(self):
        
        physical_info = []
        smoke_images = []

        for np_name in glob.glob('./data/reduced/v/*'):
            a = np.load(np_name)
            physical_info.append(a['y'])
            smoke_images.append(a['x'])
        
        self.smoke_images = smoke_images
        self.physical_info = physical_info

        tensor_x = torch.Tensor(physical_info) # transform to torch tensor
        tensor_y = torch.Tensor(smoke_images)

        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        self.train_data_loader = DataLoader(my_dataset,  batch_size=self.batch_size, shuffle=True) # create your dataloader

        print('It worked!')


    def forward(self, x):

        x = x.to(self.device)
        x = self.Linear_Layer(x)
        #I let torch figure which number to use here instead of -1
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
        x += x0 # That is the skip connection
        x = upsampler(x) #This is the by 2 upsampling step. (1, 1, 64, 48) --> (1, 1, 128, 96)

        #Fifth (and last) big block
        x = self.conv5_1(x)
        x = F.leaky_relu(x)
        x = self.conv5_2(x)
        x = F.leaky_relu(x)
        x = self.conv5_3(x)
        x = F.leaky_relu(x)
        x = self.conv5_4(x)
        #In this last big block we avoid upsampling (as the desired dimension is already attained)
        #We also avoid the skip connection

        out = self.conv_last(x)
        return out
    
    def _train(self):
        self.train()#This tells pytorch that we are in training mode

        losses = []
        x = []
        for i in range(self.epochs):
            #print('This is i!!!: ', i)
            c = 0
            x.append(i)
            for (input, label) in self.train_data_loader:

                self.optimizer.zero_grad()
                label = label.to(self.device)
                prediction = self.forward(input)
                label = torch.reshape(label, (-1, 2, 128, 96))
                #print(label.shape)
                #print(prediction.shape)
                loss = self.loss(prediction, label)
                c += loss.item()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            
            losses.append(c/len(self.smoke_images))

        plt.plot(x, losses)
        plt.savefig('result/loss.png')

#I create a folder to store some results of reconstruction of randomly
#selected images from the training set
dir_ = 'result'
if os.path.exists(dir_):
    shutil.rmtree(dir_)
# Create target directory & all intermediate directories if don't exists
os.makedirs(dir_)

my_cnn = CNN(0.0001, 400, 8)
my_cnn._train()

for k in range(20):
    r_n = np.random.randint(len(my_cnn.smoke_images))
    v_ = torch.Tensor(my_cnn.physical_info[r_n])

    my_result = my_cnn.forward(v_)
    my_result = torch.reshape(my_result, (128, 96, 2))

    to_show_gt = np.zeros((128, 96, 1))
    to_show_reconstructed = np.zeros((128, 96, 1))

    my_result = my_result.cpu()
    x_r = my_result.detach().numpy()
    x_gt = np.array(my_cnn.smoke_images[r_n])
    
    #This is a dirty way of showing images 
    for i in range(128):
        for j in range(96):
            to_show_gt[127-i][j] = 5*x_gt[i][j][1] + 3*x_gt[i][j][0]
            to_show_reconstructed[127-i][j] = 5*x_r[i][j][1] + 3*x_r[i][j][0]

    plt.imshow(to_show_gt, cmap='gray', vmin=-18, vmax=27)
    plt.savefig('result/train_gt_image_' + str(r_n) + '_gt.png')
    plt.imshow(to_show_reconstructed, cmap='gray', vmin=-18, vmax=27)
    plt.savefig('result/reconstructed_image_' + str(r_n) + '_.png')