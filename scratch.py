# %%
import numpy as np
import ThreeDLabeler as td
import nibabel as nib
# import matplotlib.pyplot as plt
#
# %%
im = nib.load('./MouseSkullData/475.mnc')
pt = td.tag_parser('MouseSkullData/475_landmarks.tag')

img = td.Image(im.get_fdata(), pt, im.header.get_zooms())

# %%
img = img.cube().scale(64)

# %%
test = img.img_transformer()


# %%
for i in test:
    i.plot()
# %%
t1 = img._cube_points()
t1t = img._square_points(t1)
td.mri_plot(img.voxels, t1t)

# %%
test = np.arange(27)
test = test.reshape(3, 3, 3)

tl = []
for i in td.images.rotations24(test):
    tl.append(i)

# %%
for i in test:
    count = 0
    for j in test:
        if np.array_equal(i.point_position, j.point_position):
            count += 1
    if count > 1:
        print(i.point_position, '\n', "was duplicated", '\n',
              j.point_position, '\n done \n')
    else:
        print("no dups")

# %%
