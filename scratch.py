# %%
import numpy as np
import ThreeDLabeler as td
import nibabel as nib 
import matplotlib.pyplot as plt
# %%
im = nib.load('./MouseSkullData/475.mnc')
pt = td.tag_parser('MouseSkullData/475_landmarks.tag')

img = td.Image(im.get_data(), im.header.get_zooms(), pt)

# %%
img_flip0 = td.Image(np.flip(im.get_data(), 0),
                            im.header.get_zooms(),
                            pt)

img_flip1 = td.Image(np.flip(im.get_data(), 1),
                            im.header.get_zooms(),
                            pt)


# We need not rotate the tag file but instead switch columns or take 
# 128-column value





# %%
for i in range(3):
    print(i)
    td.Image(np.flip(im.get_data(), i),
                            im.header.get_zooms(),
                            pt).plot()
    plt.show()


# %%

for i in range(3):
    plt.imshow(np.flip(img.voxels, i)[:, :, 300])
    plt.show()

plt.imshow(img.voxels[:, :, 300])
# %%

# %%
for i in range(3):
    plt.imshow(np.fliplr(img.voxels)[:, :, 300])
    plt.show()
# %%
plt.imshow(np.fliplr(img.voxels)[:, :, 300])
plt.show()
plt.imshow(np.flipud(img.voxels)[:, :, 300])
plt.show()
plt.imshow(img.voxels[:, :, 300])
plt.show()
# %%
for i in range(4):
    for j in range(3):
        plt.imshow(np.flip(np.rot90(img.voxels,i), j)[:, :, 300])
        plt.title(f"Rotation of: {i}, and a flip of: {j}")
        plt.show()

# %%

img2 = np.flip(img.voxels, 2)
# %%

for i in [100, 200, 228, 300, 400]:
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(img2[:, :, i])
    plt.title(f'fliped2: {i}')
    f.add_subplot(1,2,2)
    plt.imshow(img.voxels[:, :, i])
    plt.title(f'nonflipped: {i}')
    plt.show()
# %%

np.allclose(img2[:, :, 228], img.voxels[:, :, 228])

# %%
np.allclose(np.flip(np.rot90(img.voxels,3), 1)[:, :, 300],
            np.flip(np.rot90(img.voxels,1), 0)[:, :, 300])

# %%
for i in range(4):
    print(np.rot90(img.voxels, i).shape)

# %%
for i in range(3):
    for j in range(3):
        if i == j:
            pass
        else:
            plt.imshow(np.moveaxis(img.voxels, j, i)[:, :, 300])
            plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np


# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxels = cube1 | cube2 | link

# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
# colors[cube2] = 'green'

# %% and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=(colors), edgecolor='k')
plt.show()
# %%



def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

# build up the numpy logo
n_voxels = np.zeros((8, 8, 8), dtype=bool)
n_voxels[0, 0, :2] = True
n_voxels[-1, 0, :] = True
n_voxels[1, 0, 2] = True
n_voxels[2, 0, 1] = True
facecolors = np.where(n_voxels, '#FFD65DC0', '#7A88CCC0')

facecolors[0, 0, 0] = '#f4490f'
facecolors[7,7,7] = '#f40fb1'

edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
filled = np.ones(n_voxels.shape)


# personal lebeling:

# upscale the above voxel image, leaving gaps
filled_2 = explode(filled)
fcolors_2 = explode(facecolors)
ecolors_2 = explode(edgecolors)

# Shrink the gaps
x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
x[0::2, :, :] += 0.05
y[:, 0::2, :] += 0.05
z[:, :, 0::2] += 0.05
x[1::2, :, :] += 0.95
y[:, 1::2, :] += 0.95
z[:, :, 1::2] += 0.95


# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=np.transpose(fcolors_2, axes=[0,2]), edgecolors=ecolors_2)

plt.show()
# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=np.flipud(fcolors_2), edgecolors=ecolors_2)

plt.show()

# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=np.fliplr(fcolors_2), edgecolors=ecolors_2)

plt.show()

# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=(fcolors_2), edgecolors=ecolors_2)

plt.show()
# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=np.rot90(fcolors_2), edgecolors=ecolors_2)

plt.show()
# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=np.flipud(np.fliplr(fcolors_2)),
     edgecolors=ecolors_2)

plt.show()

# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=np.moveaxis(fcolors_2, 1, -1),
     edgecolors=ecolors_2)

plt.show()

# %%
