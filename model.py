import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor

class CNN(nn.Module):
    def __init__(self, lr, epochs):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
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