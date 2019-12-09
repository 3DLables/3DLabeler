# %%
import numpy as np
import ThreeDLabeler as td
import nibabel as nib
# import matplotlib.pyplot as plt
# %%
im = nib.load('./MouseSkullData/475.mnc')
pt = td.tag_parser('MouseSkullData/475_landmarks.tag')

img = td.Image(im.get_data(), im.header.get_zooms(), pt)

# %%
img = img.cube().scale(64)
# %%
test = img.rotator(90, (0, 2))
# %%
test0 = img.rotator(90, (1, 2))


# %%


# %%
img_flip0 = td.Image(np.flip(im.get_data(), 0),
                     im.header.get_zooms(),
                     pt)

img_flip1 = td.Image(np.flip(im.get_data(), 1),
                     im.header.get_zooms(),
                     pt)


# %%
t1 = img._cube_points()
t1t = img._square_points(t1)
td.mri_plot(img.voxels, t1t)

# %%
