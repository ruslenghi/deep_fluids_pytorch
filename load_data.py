import glob
import numpy as np
import os

def load_and_preprocess(velocity_dataset):

    #Decide which dataset to use to train the CNN 
    directory = None
    if velocity_dataset == 'super_reduced':
        directory = './data/super_reduced/'
    elif velocity_dataset == 'reduced':
        directory = './data/reduced/'
    elif velocity_dataset == 'full':
        directory = './data/smoke_pos21_size5_f200/'
    physical_info = []
    smoke_images = []

    #Load the info from the picked dataset, and transpose the images
    #This step was fundamental, and without it I could not attain decent visual results
    for np_name in glob.glob(os.path.join(directory,'v/*')):
        a = np.load(np_name)
        physical_info.append(a['y'])
        a_x = np.array(a['x'].transpose(2, 0, 1), dtype=np.float32)
        smoke_images.append(a_x)

    args = {}
    with open(os.path.join(directory,'args.txt'), 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            arg, arg_value = line[:-1].split(': ')
            args[arg] = arg_value
        

    #I do the preprocessing for the physical param input. The data get centered between [-1,1]
    for i in range(len(smoke_images)):
        physical_info[i][0] = (physical_info[i][0] - float(args['min_src_x_pos']))/(float(args['max_src_x_pos']) - float(args['min_src_x_pos'])) * 2 - 1
        physical_info[i][1] = (physical_info[i][1] - float(args['min_src_radius']))/(float(args['max_src_radius']) - float(args['min_src_radius'])) * 2 - 1
        #BE CAREFUL WITH THIS ONE
        physical_info[i][2] = (physical_info[i][2])/(float(args['num_frames'])) * 2 - 1

    #I preprocess the images, by dividing them by their (absolute) largest pixel value
    v_range = np.loadtxt(os.path.join(directory,'v_range.txt'))
    v_max = np.max(np.max(abs(v_range)))
    smoke_images /= v_max

    return physical_info, smoke_images, v_max