from ThreeDLabeler import preprocessing as pp
import os
import numpy as np

BASE_DIR = ('/home/kail/DataScienceProjects/skulls_new/')
TAG_DIR = ('/home/kail/DataScienceProjects/3DLabeler/txt/')

MNC_DIR = ('/run/media/kail/JPD3/ALL_MICE/Source/Original/')


tag_files = os.listdir(TAG_DIR)
mnc_files = os.listdir(MNC_DIR)

mnc_files = [i for i in mnc_files if '.mnc' in i]  # necessary for removing png
tag_files = [i for i in tag_files if '.tag' in i]


mnc_num = [i.split('.mnc')[0] for i in mnc_files]
tag_num = [i.split('_landmar')[0] for i in tag_files]

mnc_num = np.array(mnc_num)
mnc_files = np.array(mnc_files)
tag_num = np.array(tag_num)
tag_files = np.array(tag_files)

diff_files = np.unique(np.union1d(np.setdiff1d(tag_num, mnc_num),
                                  np.setdiff1d(mnc_num, tag_num)))

print('There are', diff_files.shape[0], 'files not in the same set')

# Remove all elements that are not in both folders.
for i in diff_files:
    tag_files = np.delete(tag_files, np.where(tag_num == i))
    tag_num = np.delete(tag_num, np.where(tag_num == i))
    mnc_files = np.delete(mnc_files, np.where(mnc_num == i))
    mnc_num = np.delete(mnc_num, np.where(mnc_num == i))

tag_files.sort()
tag_num.sort()
mnc_files.sort()
mnc_num.sort()

assert np.setdiff1d(tag_num, mnc_num).shape == (0,)
assert np.setdiff1d(mnc_num, tag_num).shape == (0,)
assert np.alltrue(tag_num == mnc_num)

tag_num = [i + '_landmarks.tag' for i in tag_num]
mnc_num = [i + '.mnc' for i in mnc_num]

pp.package_to_npy(BASE_DIR, mnc_files[23:],
                  tag_files[23:],
                  mnc_num[23:],
                  mnc_sub_folder=MNC_DIR,
                  tag_sub_folder=TAG_DIR,
                  output_path='/home/kail/MouseSkullsCompressed/')


