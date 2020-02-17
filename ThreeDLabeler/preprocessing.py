import numpy as np
from tqdm import tqdm
from io import StringIO
import nibabel as nib
from ThreeDLabeler.images import Image


def package_to_npy(file_path: str, mnc_files: list,
                   tag_files: list, mnc_names: list,
                   outputsize=128):
    """
    INPUT:  Path where raw image files exist,
            List of .mnc files,
            List of corresponding .tag files,
            List of .mnc prefix names

    The .mnc file is loaded
    The .tag file is parsed and converted to an ndarray via tag_parser()
    Processor class is instantiated with the .mnc and .tag file and cubes
    any images shaped as rectangular prisms and scales down image
    resolution to outputsize.

    OUTPUT: Tuple of the processed .mnc and .tag files stored as .npy file
    and saved to disk locally.
    """
    print('Starting image processing...')
    count = 0
    for i in tqdm(range(len(mnc_files))):
        img = nib.load(f'{file_path}/{mnc_files[i]}')
        tag = tag_parser(f'{file_path}/{tag_files[i]}')
        im = Image(img.get_data(), tag, img.header.get_zooms())
        im.cube().scale(outputsize)
        npy_file = (im.voxels, im.point_position)
        np.save(f'{file_path}/{mnc_names[i]}.npy', npy_file)
        count += 1

    print(f'{count} .mnc/.tag file pairs have been processed and ' +
          'saved as .npy files')


def tag_parser(file_path: str):
    """
    parses .tag files by taking the file path.
    Functionality is currently limited to only certain tag files and is not
    guaranteed
    to work everywhere
    """
    with open(file_path) as f:
        t = f.read()
        t = t.split("Points =\n")[1]
        t = t.replace(" 0.1 1 1 \"Marker\"", "")
        t = t.replace(";", "")
        t = t.replace(" \n", "\n")
        t = t[1:]
        t = StringIO(t)

        return np.genfromtxt(t, delimiter=' ')
