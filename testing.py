# %%
import nibabel as nib
import numpy as np
import importlib


# %%

img = nib.load("MouseSkullData/475.mnc")
data = img.get_data()

#%%
from ThreeDLabeler import images
from ThreeDLabeler.preprocessing import tag_parser
# importlib.reload(ThreeDLabeler.images)
tags = tag_parser('MouseSkullData/475_landmarks.tag')

test = images.Image(data, (0.035, 0.035, 0.035), tags)

print(test.point_positon)
print(test.voxels.shape)
test.cube()
print(test.point_positon)
print(test.voxels.shape)

# ############################
# # a = np.arange(125)       #
# # a = a.reshape((5, 5, 5)) #
# # print(a)                 #
# #                          #
# #                          #
# #                          #
# # # %%                     #
# ############################
# np.pad(a, (1, 3), 'constant', constant_values=(0))

# # %%

#%%
# from scipy import ndimage, misc
# import matplotlib.pyplot as plt

# array = test.voxels
# array_zoom = ndimage.zoom(array, 0.01)

test.scale(128)
print(test.point_positon)
print(test.voxels.shape)

my_array = np.loadtxt('iris_numbers.csv',delimiter=",", skiprows=1)

tag_parse

