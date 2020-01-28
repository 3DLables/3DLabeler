
import numpy as np
from tqdm import tqdm
import os
from scipy.ndimage import rotate
import random
from random import shuffle
import ThreeDLabeler as td
from ThreeDLabeler.images import Image

import sys


basepath = '/Users/michaeldac/Code/CUNY/698/ReducedData64/'
file_list = os.listdir(basepath)
file_list = [i for i in file_list if i.endswith('.npy')]

skulls = []
print('LOADING .npy FILES...')
for file in tqdm(file_list): # -1 because of the model
    file = np.load(basepath + file, allow_pickle=True)
    file = td.Image(file[0], file[1])
    skulls.append(file)
    
#skulls = np.array(skulls)

print(f'index value type: {type(skulls[0])}')

def rotate_skulls(skulls):
    skull_list = []
    for i in tqdm(skulls):
        skull_list.append(i.img_transformer())
    skull_list = np.array(skull_list).flatten()

    skull_list = np.array(skull_list)
    return skull_list

augmented_skulls = rotate_skulls(skulls)


print (f'length of augmented data: {len(augmented_skulls)}')

file_path = '/Users/michaeldac/Code/CUNY/698/AugmentedData'
count = 1
for i in tqdm(augmented_skulls):
    npy_file = (i.voxels, i.point_position)
    np.save(f'{file_path}/skull_{count}.npy', npy_file)
    count += 1