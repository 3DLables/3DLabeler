import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


class Image:
    """
    Processor class for annotating 3D scans.
    Arguments:
    voxels: a 3D numpy array
    voxel_size: a tuple/list of three numbers indicating the voxel size in mm,
    cm etc point_position: the position in 3D of each point of interest.
    See tag_parser for more info
    """

    def __init__(self, voxels, voxel_size, point_position):
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
        """
        Scales an cubic image to a certain number of voxels.
        This function relies on numpy's ndimage.zoom function
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
            ax.append(fig.add_subplot(rows, columns, i+1))
            ax[-1].set_title("Image depth: "+str(im_slice))  # set title
            plt.imshow(im)
            plot_cols = np.array([0, 1, 2])
            plot_cols = plot_cols[plot_cols != vcol]
            plt.plot(points[i,
                            min(plot_cols)],
                     points[i,
                            max(plot_cols)],
                     'ro')

        plt.show()

    def _cube_points(self):
        """cubes the point positions for rotation"""

        cubedims = self.voxels.shape
        points = self.point_position
        points = np.rint(points).astype('int')

        arr = np.zeros((cubedims), dtype=int)  # creates empty array
        for i in range(self.point_position.shape[0]):
            arr[points[i, 0], points[i, 1], points[i, 2]] = i+1
            # +1 Avoide zeros
        return arr

    def _square_points(self, arr):
        """Takes a cubed array and returns it in square format
        Note that it does not affect `self` so this has to be passed to
        self in the rotate function"""

        flatpoints = np.zeros((self.point_position.shape), dtype=int)
        # double (()) to make it a tuple
        npoints = self.point_position.shape[0]

        for i in range(npoints):
            flatpoints[i, :] = np.where(arr == i+1)

        return flatpoints

    def rotator(self, angle, axes):

        voxels = self.voxels
        voxels = rotate(voxels, angle=angle, axes=axes)
        # self.voxels = voxels

        # TODO Why does this flip the opposite way?
        # Probably because my columns are in the wrong order
        points = self._cube_points()
        points = rotate(points, angle=angle, axes=axes)
        points = self._square_points(points)
        # self.point_position = pointstpoints

        return Image(voxels, 1, points)
        # TODO why can't I just return self?

# TODO Add possibility to not just cube an image
# TODO Add Storeage/writing functionality


def _flipud(m):
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return m[::-1, ...]


def _fliplr(m):
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return m[:, ::-1]


def _flipbf(m):
    if m.ndim < 3:
        raise ValueError("Input must be >= 3-d.")
    return m[:, :, ::-1]
