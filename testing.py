# %%
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
import numpy as np

# %%

img = nib.load("MouseSkullData/475.mnc")
data = img.get_data()

#%%
from ThreeDLabeler import images
from ThreeDLabeler.PreProcessing import tag_parser

tags = tag_parser('MouseSkullData/475_landmarks.tag')

test = images.Image(data, 'voxelsize', tags)



a = np.arange(125)
a = a.reshape((5, 5, 5))
print(a)



# %%
np.pad(a, (1, 3), 'constant', constant_values=(0))

# %%

#%%
