import numpy as np

class Image:
    '''the image class for 3d labels'''

    def __init__(self, voxels, voxel_size, point_positon):
        self.voxels = voxels
        self.voxel_size = voxel_size
        self.point_positon = point_positon / voxel_size
        self.dimensions = {}
        self.dimensions['x'] = self.voxels.shape[0]
        self.dimensions['y'] = self.voxels.shape[1]
        self.dimensions['z'] = self.voxels.shape[2]
        # Todo add padding and scalling

    def pad(self, target_dims):
        # TODO 
        
        pass


    def cube(self):
        """Returns a cube image with all dimensions equal to the longest."""

        max_dim = max(self.dimensions, key = self.dimensions.get)
        max_dim = self.dimensions[max_dim]
        x_target = (max_dim - self.dimensions['x']) / 2
        y_target = (max_dim - self.dimensions['y']) / 2
        z_target = (max_dim - self.dimensions['z']) / 2

        self.voxels = np.pad(self.voxels,
                            ((int(np.ceil(x_target)), int(np.floor(x_target))),
                             (int(np.ceil(y_target)), int(np.floor(y_target))),
                             (int(np.ceil(z_target)), int(np.floor(z_target)))),
                            'constant',
                            constant_values=(0))
        self.point_positon = self.point_positon + [np.ceil(x_target),
                                                     np.ceil(y_target),
                                                     np.ceil(z_target)]

        

    def scale(self, size):
        # TODO
        pass

def ratio(dims):
    """Calculates the ratio of n numbers"""
    dims = np.asarray(dims)
    total = dims.sum()
    
    return dims/total


