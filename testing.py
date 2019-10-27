# %%
import nibabel as nib
import numpy as np
import importlib


# %%

img = nib.load("MouseSkullData/475.mnc")
data = img.get_data()

#%%
from ThreeDLabeler import images
from ThreeDLabeler.PreProcessing import tag_parser
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
