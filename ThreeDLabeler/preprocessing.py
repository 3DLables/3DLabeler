import numpy as np
from tqdm import tqdm
from io import StringIO
import os


def package_to_npy(file_path: str, mnc_files: list,
                   tag_files: list, mnc_names: list):
    """
    INPUT:  Path where raw image files exist,
            List of .mnc files,
            List of corresponding .tag files,
            List of .mnc prefix names

    The .mnc file is loaded
    The .tag file is parsed and converted to an ndarray via tag_parser()
    Processor class is instantiated with the .mnc and .tag file and cubes
    any images shaped as rectangular prisms and scales down image
    resolution to 128x128x128.

    OUTPUT: Tuple of the processed .mnc and .tag files stored as .npy file
    and saved to disk locally.
    """
    print('Starting image processing...')
    count = 0
    for i in tqdm(range(len(mnc_files))):
        img = nib.load(f'{file_path}/{mnc_files[i]}')
        tag = tag_parser(f'{file_path}/{tag_files[i]}')
        im = Processor(img.get_data(), img.header.get_zooms(), tag)
        im.cube().scale(128)
        npy_file = (im.voxels, im.point_position)
        np.save(f'{file_path}/{mnc_names[i]}.npy', npy_file)
        count += 1

    print(f'{count} .mnc/.tag file pairs have been processed and ' +
          'saved as .npy files')


def upload_to_gcp(path_to_files: str, project_name: str, bucket_name: str):
    """
    INPUT:  Path where processed images exist,
            GCP project name,
            GCP bucket name

    GCP client id'd and blob located.
    Loop through folder of processed images and upload one at a time.

    OUTPUT: Returns None. Processed image files are uploaded to GCP
    Cloud Storage
    """
    print('Starting upload to Google Cloud Storage project')
    #  storage_client = storage.Client(project=project_name)
    #  bucket = storage_client.get_bucket(bucket_name)

    count = 0
    for filename in tqdm(os.listdir(path_to_files)):
        blob = bucket.blob(filename)
        blob.upload_from_filename(f'{path_to_files}/{filename}')
        # print(f'{filename} successfully uploaded to {bucket_name} bucket.')
        count += 1

    print(f'{count} blobs were uploaded to Project:{project_name}, Bucket:{bucket_name}')


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
