from ThreeDLabeler import preprocessing as pp
import os
import numpy as np
SKULL_DIR = r"G:\MouseSkulls"
file_list = os.listdir(SKULL_DIR)

mnc_files = [i for i in file_list if '.mnc' in i]
tag_files = [i for i in file_list if '.tag' in i]
file_names = ['skull_' + i.split('.')[0] for i in mnc_files]  # change this one

existing = [i for i in file_list if '.npy' in i]
existing = [i.split('.npy')[0] for i in existing]
existing = [i[6:] for i in existing]


file_names.sort()
mnc_files.sort()
tag_files.sort()
existing.sort()

mnc_files = np.array(mnc_files)
tag_files = np.array(tag_files)


tag_test = np.array([i.split('_landmarks.t')[0] for i in tag_files])
mnc_test = np.array([i.split('.')[0] for i in mnc_files])
existing = np.array(existing)
file_names = np.array(file_names)

processed_files = np.isin(tag_test, existing)

mnc_files = mnc_files[~processed_files]
tag_files = tag_files[~processed_files]
file_names = file_names[~processed_files]

pp.package_to_npy(SKULL_DIR, mnc_files[3:], tag_files[3:], file_names[3:])
