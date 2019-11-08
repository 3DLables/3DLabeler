import numpy as np
from scipy import ndimage

class Image:
    """Image class for annotating 3D scans.
    Arguments: 
    voxels: a 3D numpy array
    voxel_size: a tuple/list of three numbers indicating the voxel size in mm, cm etc
    point_position: the position in 3D of each point of interest. See tag_parser for more info
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
                            ((int(np.ceil(x_target)), int(np.floor(x_target))),
                             (int(np.ceil(y_target)), int(np.floor(y_target))),
                             (int(np.ceil(z_target)), int(np.floor(z_target)))),
                            'constant',
                            constant_values=(0))
        self.point_position = self.point_position + [np.ceil(z_target),
                                                     np.ceil(y_target),
                                                     np.ceil(x_target)]


        
    def scale(self, size=128):
       """Scales an cubic image to a certain number of voxels.
       This function relies on numpy's ndimage.zoom function"""
       scale_factor = size / max(self.voxels.shape)
       self.voxels = ndimage.zoom(self.voxels, scale_factor)
       self.point_position = self.point_position * scale_factor
       self.voxel_size = False # To ignore this
       pass

# TODO Add posibility to not just cube an image
# TODO Add Storeage/writing functionality

