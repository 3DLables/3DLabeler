import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


class Image:
    """
    Processor class for annotating 3D scans.
    Arguments:
    voxels: a 3D numpy array
    voxel_size: a tuple/list of three numbers indicating the voxel size in mm,
    cm etc point_position: the position in 3D of each point of interest.
    See tag_parser for more info
    """

    def __init__(self, voxels, point_position, voxel_size=(1, 1, 1)):
        self.voxels = voxels
        self.voxel_size = voxel_size
        self.point_position = point_position / voxel_size

    def cube(self):
        """Returns a cube image with all dimensions equal to the longest."""

        dims = self.voxels.shape
        max_dim = max(dims)

        x_target = (max_dim - dims[0]) / 2
        y_target = (max_dim - dims[1]) / 2
        z_target = (max_dim - dims[2]) / 2

        self.voxels = np.pad(self.voxels,
                             ((int(np.ceil(x_target)),
                               int(np.floor(x_target))),
                              (int(np.ceil(y_target)),
                               int(np.floor(y_target))),
                              (int(np.ceil(z_target)),
                               int(np.floor(z_target)))),
                             'constant',
                             constant_values=(0))

        self.point_position = self.point_position + [np.ceil(z_target),
                                                     np.ceil(y_target),
                                                     np.ceil(x_target)]

        return(self)

    def scale(self, size=128):
        """Scales a 3D Numpy array to the desired dimensions

        Keyword Arguments:
            size {int} -- The number of pixels that should be scaled too.
            It uses the numpy.ndimage scale function. (default: {128})
        """
        scale_factor = size / max(self.voxels.shape)
        self.voxels = ndimage.zoom(self.voxels, scale_factor)
        self.point_position = self.point_position * scale_factor
        self.voxel_size = False  # To ignore this

        return(self)

    def plot(self, vcol=1):
        """
        Graphs an points. pt_cols is used to set the cols to iterate
        over (different views)
        """
        img = self.voxels
        points = self.point_position
        ax = []
        fig = plt.figure(figsize=(9, 8))
        # TODO make this setable in the function call
        columns = 3
        rows = 2

        for i in range(points.shape[0]):
            im_slice = int(np.round(points[i, vcol]))
            if vcol == 0:
                im = img[im_slice, :, :]
            elif vcol == 1:
                im = img[:, im_slice, :]
            else:
                im = img[:, :, im_slice]
            plot_cols = np.array([0, 1, 2])
            plot_cols = plot_cols[plot_cols != vcol]
            ax.append(fig.add_subplot(rows, columns, i+1))  # set title
            ax[-1].set_title("Image depth: " +
                             str(im_slice) +
                             '\n x-axis' +
                             str(np.round(points[i, min(plot_cols)])) +
                             '\n y-axis' +
                             str(np.round(points[i, max(plot_cols)])))
            plt.imshow(im)

            plt.plot(points[i, min(plot_cols)],
                     points[i, max(plot_cols)],
                     'ro')
        # TODO Fix bug where points don't plot properly
        # BUG
        plt.show()

    def _cube_points(self):
        """Cubes two dimensional array of key points for rotation

        Returns:
            numpy array -- A 3D Numpy array with point 1-n shown in 3D Space
        """

        cubedims = self.voxels.shape
        points = self.point_position
        points = np.rint(points).astype('int')  # sets to int for indexing

        arr = np.zeros((cubedims), dtype=int)  # creates empty array
        for i in range(self.point_position.shape[0]):
            arr[points[i, 0], points[i, 1], points[i, 2]] = i+1
            # +1 Avoide zeros
        return arr

    def _square_points(self, arr):
        """Takes a cubed point array and return a 2D version of it with key
        points

        Arguments:
            arr {3D Numpy Array} -- The 3D array with key points in it

        Returns:
            2D Numpy Array -- The x, y, z coordinates of each key point

        Yields:
            numpy array -- when used as an iterator. This may be a bug
        """
        flatpoints = np.zeros((self.point_position.shape), dtype=int)
        # double (()) to make it a tuple
        npoints = flatpoints.shape[0]

        for i in range(npoints):
            flatpoints[i, :] = np.where(arr == i+1)

        return flatpoints

    def img_transformer(self, number_rot=24):
        """Generates 24 projections of a 3D image along with the key points

        Returns:
            ThreeDLabeler.Image -- The voxels and key points for a projection
            of the image
        """
        voxels = []
        points = []
        if number_rot == 24:
            rot_fun = rotations24
        elif number_rot == 4:
            rot_fun = rotations4
        elif number_rot == 2:
            rot_fun = rotations2
        else:
            raise ValueError("Incorrect number or rotations, try 4, 24")
        for i in rot_fun(self.voxels):
            voxels.append(i)
        for j in rot_fun(self._cube_points()):
            points.append(self._square_points(j))

        imgs = []
        for i in range(number_rot):
            imgs.append(Image(voxels[i], points[i]))

        return imgs


# TODO Add possibility to not just cube an image
# TODO Add Storeage/writing functionality
def rotations24(polycube):
    """https://stackoverflow.com/
    questions/33190042/how-to-calculate-all-24-rotations-of-3d-array"""
    # imagine shape is pointing in axis 0 (up)

    # 4 rotations about axis 0
    yield from rotations_flip4(polycube, 0)

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    yield from rotations_flip4(rot90(polycube, 2, axis=1), 0)

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    yield from rotations_flip4(rot90(polycube, axis=1), 2)
    yield from rotations_flip4(rot90(polycube, -1, axis=1), 2)

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    yield from rotations_flip4(rot90(polycube, axis=2), 1)
    yield from rotations_flip4(rot90(polycube, -1, axis=2), 1)


def rotations4(polycube):
    yield polycube  # Unit yeld
    yield polycube[::-1, :, :]
    yield polycube[:, ::-1, :]
    yield polycube[..., ::-1]


def rotations2(polycube):
    yield polycube  # Unit yeld
    yield polycube[::-1, :, :]
    # yield polycube[:, ::-1, :]
    # yield polycube[..., ::-1]


def rotations_flip4(polycube, axis):
    """List the four rotations of the given cube about the given axis."""
    for i in range(4):
        yield rot90(polycube, i, axis)


def rot90(m, k=1, axis=2):
    """Rotate an array k*90 degrees in the counter-clockwise direction
    around the given axis
    This differs from np's rot90 because it's 3D
    """
    m = np.swapaxes(m, 2, axis)
    m = np.rot90(m, k)
    m = np.swapaxes(m, 2, axis)
    return m
