# %%
from scipy.ndimage import rotate
import numpy as np
import ThreeDLabeler as td
import nibabel as nib
import matplotlib.pyplot as plt
# %%
im = nib.load('./MouseSkullData/475.mnc')
pt = td.tag_parser('MouseSkullData/475_landmarks.tag')

img = td.Image(im.get_data(), im.header.get_zooms(), pt)

# %%
img = img.cube().scale(64)

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
        plt.imshow(np.flip(np.rot90(img.voxels, i), j)[:, :, 300])
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
    f.add_subplot(1, 2, 2)
    plt.imshow(img.voxels[:, :, i])
    plt.title(f'nonflipped: {i}')
    plt.show()
# %%

np.allclose(img2[:, :, 228], img.voxels[:, :, 228])

# %%
np.allclose(np.flip(np.rot90(img.voxels, 3), 1)[:, :, 300],
            np.flip(np.rot90(img.voxels, 1), 0)[:, :, 300])

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


# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between
# them
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

facecolors[0, 0, 0] = '#FF0000'
facecolors[0, 0, 7] = '#00FF00'
facecolors[0, 7, 7] = '#0000FF'
facecolors[7, 7, 7] = '#FFFF00'

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

def cube_plotter(fcols):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(x, y, z, filled_2, facecolors=fcols, edgecolors=ecolors_2)

    plt.show()


# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=np.flipud(
    fcolors_2), edgecolors=ecolors_2)

plt.show()

# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=np.fliplr(
    fcolors_2), edgecolors=ecolors_2)

plt.show()

# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=(fcolors_2), edgecolors=ecolors_2)

plt.show()
# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=np.rot90(
    fcolors_2), edgecolors=ecolors_2)

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


# %%
test = np.array([i for i in range(27)])

# %%
test = test.reshape(3, 3, 3)

# %%
np.moveaxis(test, 1, 2)

# %%
np.moveaxis(test, 0, 1)

# %%
np.flipud(test)

# %%
np.fliplr(test)


# %%
np.flip(test)

# %%

print("original\n", test)
print("axis0\n", test[::-1, :, :])
print("axis1\n", test[:, ::-1, :])
print("axis2\n", test[:, :, ::-1])


# %%


# from numpy import flipud, fliplr # we no longer want this
# %%


def flipud(m):
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return m[::-1, ...]


def fliplr(m):
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return m[:, ::-1]


def flipbf(m):
    if m.ndim < 3:
        raise ValueError("Input must be >= 3-d.")
    return m[:, :, ::-1]


def rotate3d(m, k, f):
    k = k % 4
    if k == 0:
        return f(m)
    elif k == 1:
        return f(m).swapaxes(0, 1)
    elif k == 2:
        return f(m).swapaxes(0, 2)
    else:
        pass
# %%


flip_rot = []
arr = []

for i in range(3):
    flip_rot.append('original_rotation' + str(i))
    arr.append(np.rot90(test, k=i))
    flip_rot.append('fliplr_rotation' + str(i))
    arr.append(np.rot90(fliplr(test), k=i))
    flip_rot.append('flipbf_rotation' + str(i))
    arr.append(np.rot90(flipbf(test), k=i))
    flip_rot.append('flipud_rotation' + str(i))
    arr.append(np.rot90(flipud(test), k=i))


# %%
for i in range(3):
    for j in range(3):
        print(np.alltrue(fliplr(flipud(test)) == fliplr(test).swapaxes(i, j)))

# %%

funlist = [fliplr, flipud, flipbf]


def swapper12(m, funlist):
    array_list = []
    for f in funlist:
        for i in (range(3)):
            array_list.append(np.rot90(f(m)))
    for i in range(3):
        array_list.append(np.rot90(m, i))

    return array_list


# %%
teest = swapper12(test, funlist)

# %%

tlist = []
for i in range(3):
    for j in range(4):
        tlist.append(np.rollaxis(test, i, j))

# %%
tlist = []
for i in [0, 90, 180, 270]:
    for j in range(3):
        for k in range(3):
            if j == k:
                pass
            else:
                print(i, j, k)
                t = rotate(test, i, axes=(j, k))
                tlist.append(t)


# %%
rotate(test, 90, (2, 1))

# %%

# %%
len(tlist) == len(set(tlist))

# %%
tlist = np.array(tlist)

# %%
np.unique(tlist)

# %%
for i in range(len(tlist)):
    print(i+1)
    if (tlist[i] in tlist[:i] or tlist[i] in tlist[(i+1):]):
        print("the same")
    else:
        print('differetn')

# %%


for i in [0, 90, 180, 270]:
    for j in range(3):
        for k in range(3):
            if j == k:
                pass
            else:
                t = rotate(img.voxels, i, axes=(j, k))
                td.mri_plot(t, pt)


# %%

betterList = tlist[0].tolist()

# %%
betterList = [i.tolist() for i in tlist]
# %%

imscale = img.cube().scale(128)
s = imscale.voxels
kp = imscale.point_position

# s = skulls[0]

# %% this works
a, b, c = (2, 1, 0)
s90 = rotate(s, 90, (0, 2))
s180 = rotate(s, 180, (0, 2))
s270 = rotate(s, 270, (0, 2))
kp90 = np.array([[i[a], i[b], 128-i[c]] for i in kp])
kp180 = np.array([[i[a], i[b], 128-i[c]] for i in kp90])
kp270 = np.array([[i[a], i[b], 128-i[c]] for i in kp180])
td.mri_plot(s, kp, vcol=1)
td.mri_plot(s90, kp90, vcol=1)
td.mri_plot(s180, kp180, vcol=1)
td.mri_plot(s270, kp270, vcol=1)

# %% this works
a, b, c = (1, 0, 2)
s90 = rotate(s, 90, (1, 2))
s180 = rotate(s, 180, (1, 2))
s270 = rotate(s, 270, (1, 2))
kp90 = np.array([[i[a], 128-i[b], i[c]] for i in kp])
kp180 = np.array([[i[a], 128-i[b], i[c]] for i in kp90])
kp270 = np.array([[i[a], 128-i[b], i[c]] for i in kp180])
# td.mri_plot(s, kp, vcol=1)
td.mri_plot(s90, kp90, vcol=1)
td.mri_plot(s180, kp180, vcol=1)
td.mri_plot(s270, kp270, vcol=1)

# %%
a, b, c = (1, 0, 2)
s90 = rotate(s, 90,  (0, 1))
s180 = rotate(s, 180, (0, 1))
s270 = rotate(s, 270, (0, 1))
kp90 = np.array([[i[a], 128-i[b], i[c]] for i in kp])
kp180 = np.array([[i[a], 128-i[b], i[c]] for i in kp90])
kp270 = np.array([[i[a], 128-i[b], i[c]] for i in kp180])
# td.mri_plot(s, kp, vcol=1)
td.mri_plot(s90, kp90, vcol=1)
td.mri_plot(s180, kp180, vcol=1)
td.mri_plot(s270, kp270, vcol=1)

# %%
imscale.point_position

# %%
intkp = np.rint(imscale.point_position).astype('int')

arr = np.empty((128, 128, 128), dtype=object)


for i in range(intkp.shape[0]):
    arr[intkp[i][0], intkp[i][1], intkp[i][2]] = 'point_' + str(i)
# %%


arr2 = np.fliplr(arr).swapaxes(0, 1)

# %%
