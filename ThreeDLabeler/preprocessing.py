# Data science tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Image processing
import nibabel as nib
from scipy import ndimage

# Base operations
from tqdm import tqdm
from io import StringIO
import os

# Cloud interface
from google.cloud import storage
credential_path = "/Users/michaeldac/Downloads/mouse-labeler-cff0443f5b5e.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

GCP_PROJECT_NAME = 'mouse-labeler'
GCP_BUCKET_NAME =  'skull-images'

RAW_IMAGE_DIRECTORY = '/Users/michaeldac/Code/CUNY/698/Skulls'
PROCESSED_IMAGE_DIRECTORY = '/Users/michaeldac/Code/CUNY/698/ReducedSkulls'

skulls_folder = os.listdir(RAW_IMAGE_DIRECTORY)

def main():

    # fetch and sort the .mnc and .tag files
    mnc_files = [f for f in skulls_folder if 'mnc' in f]
    tag_files = [f for f in skulls_folder if 'tag' in f]
    mnc_names = [i.split('.mnc')[0] for i in mnc_files]
        
    mnc_files.sort()
    tag_files.sort()
    mnc_names.sort()

    package_to_npy(skulls_folder, mnc_files, tag_files, mnc_names)

    upload_to_gcp(PROCESSED_IMAGE_DIRECTORY, GCP_PROJECT_NAME, GCP_BUCKET_NAME)
   





def package_to_npy(folder_list, mnc_files, tag_files, mnc_names):
    """
    INPUT:  List of .mnc files, 
            List of corresponding .tag files, 
            List of .mnc prefix names
    
    The .mnc file is loaded 
    The .tag file is parsed and converted to an ndarray via tag_parser()
    Image class is instantiated with the .mnc and .tag file and cubes
    any images shaped as rectangular prisms and scales down image 
    resolution to 128x128x128.
    
    OUTPUT: Tuple of the processed .mnc and .tag files stored as .npy file 
    and saved to disk locally.
    """
    
    count = 0
    for i in tqdm(range(len(mnc_files))):
        img = nib.load(f'{folder_list}/{mnc_files[i]}')
        tag = tag_parser(f'{folder_list}/{tag_files[i]}')
        im = Image(img.get_data(), (0.035, 0.035, 0.035), tag)
        im.cube().scale(128)
        npy_file = (im.voxels, im.point_position)
        np.save(f'{PROCESSED_IMAGE_DIRECTORY}/{mnc_names[i]}.npy', npy_file)
        count += 1
    
    print('f{count} .mnc/.tag file pairs have been saved as .npy files')


def upload_to_gcp(path_to_files, project_name, bucket_name ):
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.get_bucket(bucket_name)
    
    count = 0
    for filename in os.listdir(path_to_files):
        blob = bucket.blob(filename)
        blob.upload_from_filename(path_to_files + filename)
        print(f'{filename} successfully uploaded to {bucket_name} bucket.')
        count += 1

    print(f'{count} blobs were uploaded to Project:{project_name}, Bucket:{bucket_name}')


def tag_parser(file_path):
    """parses .tag files by taking the file path. 
    Functionality is currently limited to only certain tag files and is not guaranteeded 
    to work everywhere"""
    with open(file_path) as f:
        t = f.read()
        t = t.split("Points =\n")[1]
        t = t.replace(" 0.1 1 1 \"Marker\"", "")
        t = t.replace(";", "")
        t = t.replace(" \n", "\n")
        t = t[1:]
        t = StringIO(t)

        return np.genfromtxt(t, delimiter=' ')

def mri_point_plot(img, points, vcol=1):
    """Graphs an points. pt_cols is used to set the cols to iterate 
    over (different views)
    """
    
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
        ax.append( fig.add_subplot(rows, columns, i+1))
        ax[-1].set_title("Image depth: "+str(im_slice))  # set title
        plt.imshow(im)
        plot_cols = np.array([0, 1, 2])
        plot_cols = plot_cols[plot_cols != vcol]
        plt.plot(points[i, min(plot_cols)], points[i, max(plot_cols)], 'ro')

    plt.show()


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

        return(self)
        
    def scale(self, size=128):
        """Scales an cubic image to a certain number of voxels.
       This function relies on numpy's ndimage.zoom function"""
        scale_factor = size / max(self.voxels.shape)
        self.voxels = ndimage.zoom(self.voxels, scale_factor)
        self.point_position = self.point_position * scale_factor
        self.voxel_size = False # To ignore this
        
        return(self)

# TODO Add posibility to not just cube an image
# TODO Add Storeage/writing functionality

if __name__=="__main__":
    main()