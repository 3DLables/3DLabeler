import numpy as np

class Image:
    '''the image class for 3d labels'''

    def __init__(self, voxels, voxel_size, point_positon):
        self.voxels = voxels
        self.voxel_size = voxel_size
        self.point_positon = point_positon
        self.dimensions = {}
        self.dimensions['x'] = self.voxels.shape[0]
        self.dimensions['y'] = self.voxels.shape[1]
        self.dimensions['z'] = self.voxels.shape[2]
        # Todo add padding and scalling

    def pad(self, target_dims):
        # TODO 
        
        pass


    def scale(self, size):
        # TODO
        pass

def ratio(dims):
    """Calculates the ratio of n numbers"""
    dims = np.asarray(dims)
    total = dims.sum()
    
    return dims/total
