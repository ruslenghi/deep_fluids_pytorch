import glob
import numpy as np
import os
import config
import torch

def load_and_preprocess(my_list):

    print(os.path.dirname(os.path.realpath(__file__)))

    directory = os.path.dirname(os.path.realpath(__file__)) + "\\" + config.DATASET

    #Insert the directory containing the v folder, with the velocity fields
    #if torch.cuda.is_available():
    #    directory = '/cluster/home/ruslenghi/data/' + config.DATASET
    #else:
    #    directory = '/Users/Ricca/Desktop/data/' + config.DATASET

    physical_info,  smoke_images, to_load = [], [], []
    np.random.seed(1)

    counter = -1
    for np_name in glob.glob(os.path.join(directory,'v/*')):
        #print(np_name)
        counter += 1

    if config.N_IMAGES == -1: n_images = counter
    else: n_images = min(counter, config.N_IMAGES)

    k = 0
    while k < n_images:
        
        if n_images >= counter:
            to_load.append(k)
            k += 1

        else:
            r = np.random.randint(counter)
            if r in to_load:
                pass
            else:
                to_load.append(r)
                k += 1

    counter = -1
    for np_name in glob.glob(os.path.join(directory,'v/*')):
        counter += 1
        if config.TRAIN == False and counter in to_load or config.TRAIN == True:
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
        
        if args['min_src_x_pos'] != args['max_src_x_pos']:
            physical_info[i][0] = (physical_info[i][0] - float(args['min_src_x_pos']))/(float(args['max_src_x_pos']) - float(args['min_src_x_pos'])) * 2 - 1
        else:
            physical_info[i][0] = 0
        
        if args['min_src_radius'] != args['max_src_radius']:
            physical_info[i][1] = (physical_info[i][1] - float(args['min_src_radius']))/(float(args['max_src_radius']) - float(args['min_src_radius'])) * 2 - 1
        else:
            physical_info[i][1] = 0
        
        if args['min_frames'] != args['max_frames']:
            physical_info[i][2] = (physical_info[i][2] - float(args['min_frames']))/(float(args['max_frames']) - float(args['min_frames'])) * 2 - 1
        else:
            physical_info[i][2] = 0

    #I preprocess the images, by dividing them by their (absolute) largest pixel value
    v_range = np.loadtxt(os.path.join(directory,'v_range.txt'))
    v_max = np.max(abs(v_range))
    smoke_images /= v_max

    PE_physical_info = []

    #In case we want to use positional encoding
    if config.PE == True:

        for i in range(len(physical_info)):

            a_y = physical_info[i]

            my_y, counter = a_y, 0
            for i in my_list:
                if i != 0:
                    new_y = np.concatenate((np.sin(2**counter*np.pi*a_y), np.cos(2**counter*np.pi*a_y)), axis=0)
                    my_y = np.concatenate((my_y, new_y), axis = 0)
                counter += 1

            PE_physical_info.append(my_y)

    PE_physical_info = np.array(PE_physical_info)

    #PE_physical_info.shape = 3 + 6 * (numero di potenze non nulle)
    #Nel nostro caso easy PE_physical_info.shape = 21

    if config.PE == True:
        return PE_physical_info, smoke_images

    else:
        return physical_info, smoke_images