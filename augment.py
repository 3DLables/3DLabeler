
import numpy as np
from tqdm import tqdm
import os
from scipy.ndimage import rotate
import random
from random import shuffle
import ThreeDLabeler.images as td
import sys


basepath = '/Users/michaeldac/Code/CUNY/698/ReducedData/'
file_list = os.listdir(basepath)
file_list = [i for i in file_list if i.endswith('.npy')]

voxels = []
points = []
print('LOADING .npy FILES...')
for file in tqdm(file_list): # -1 because of the model
    file = np.load(basepath+file, allow_pickle=True)
    voxels.append(file[0])
    points.append(file[1])

print('CONVERTING LIST TO NDARRAY...')
points = np.array(points)
voxels = np.array(voxels)

skulls = []
print('REDUCING IMAGES...')
for i in tqdm(range(len(file_list))):
    skull = td.Image(voxels[i], points[i])
    skull = skull.scale(64)
    skulls.append(skull)

file_path = '/Users/michaeldac/Code/CUNY/698/ReducedData64'
count = 1
for i in tqdm(skulls):
    npy_file = (i.voxels, i.point_position)
    np.save(f'{file_path}/skull_{count}.npy', npy_file)
    count += 1






print (len(skulls))