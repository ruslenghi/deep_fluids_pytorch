import torch

#Be careful, STL and PROGRESSIVE cannot be both True
TRAIN = True
#Keep in mind that plotting the gradients of the layers makes runtime about 50% longer
GAN = False
PROGRESSIVE = False
STL = False

PE = True #Positional Encoding
if PE == False:
    POWER_LIST = []
else:
    POWER_LIST = [1, 1, 1, 0, 1]

GRADIENT = True
LR_MAX = 1e-4
if GRADIENT:
    LR_MIN = 1e-4
else:
    LR_MIN = 2.5e-6

DATASET = 'super_reduced'# smoke_pos21_size5_f200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
EPOCHS = 50 #240
BATCH_SIZE = 8
CHANNELS_IMG = 2
Z_DIM = 3
NUM_WORKERS = 4

STL_WEIGHTS = [1, 1, 1, 1, 0.25]
GAN_WEIGHT = 0.005
NUM_CHANNELS = 128
NUM_BIG_BLOCKS = 4 #4
CONV_LAYERS_PER_BB = 4 #4
BETA_1 = 0.5
BETA_2 = 0.999

Y_RES = 128
X_RES = 96
N_IMAGES = 200 #200 #Set to -1 to print the reconstruction of the whole dataset