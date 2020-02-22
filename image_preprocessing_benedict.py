from ThreeDLabeler import preprocessing as pp
import os
import numpy as np

BASE_DIR = ('/mnt/Storage2/Hallgrimsson/Automated_Phenotyping/Global/' +
            'ALL_MICE/Source/')
TAG_DIR = ("/mnt/Storage2/Hallgrimsson/Automated_Phenotyping/Global/ALL_MICE" +
           '/Source/txt')

MNC_DIR = ("/mnt/Storage2/Hallgrimsson/Automated_Phenotyping/Global/ALL_MICE" +
           '/Source/MNC')

tag_files = os.listdir(TAG_DIR)
mnc_files = os.listdir(MNC_DIR)


mnc_files = [i for i in mnc_files if '.mnc' in i]  # necessary for removing png
tag_files = [i for i in tag_files if '.tag' in i]


mnc_num = [i.split('_to_Global')[0] for i in mnc_files]
tag_num = [i.split('_landmar')[0] for i in tag_files]

mnc_num = np.array(mnc_num)
mnc_files = np.array(mnc_files)
tag_num = np.array(tag_num)
tag_files = np.array(tag_files)

diff_files = np.setdiff1d(tag_num, mnc_num)

assert diff_files.shape == (1,)  # to make sure there aren't more than one.

tag_files = np.delete(tag_files, np.where(tag_num == diff_files[0]))
tag_num = np.delete(tag_num, np.where(tag_num == diff_files[0]))

assert np.alltrue(tag_num == mnc_num)

pp.package_to_npy(SKULL_DIR, mnc_files[:3], tag_files[:3], mnc_num[:3],
                    )
