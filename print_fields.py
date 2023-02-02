import torch
import sys
import shutil

from utils import *
from load_data import *
from model import *

config.TRAIN = False

#Write the name of the model you want to load when you launch the program
filename = sys.argv[1]

sum = 0
for i in config.POWER_LIST:
    sum += i

physical_info, smoke_images = load_and_preprocess(config.POWER_LIST)

dir_ = 'result'
if os.path.exists(dir_):
    shutil.rmtree(dir_)
os.makedirs(dir_)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Generator2d(STL=config.STL, GAN=config.GAN, PROGRESSIVE=config.PROGRESSIVE, sum=sum)
model = torch.load('my_models/' + filename + '.pt', map_location=device)
model.eval()

print_images(model, smoke_images, physical_info)
