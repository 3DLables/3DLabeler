import os
files = os.listdir('MouseSkullData/')
files.sort()


mnc_files = [f for f in files if 'mnc' in f]

tag_files = [f for f in files if 'tag' in f]


# [i.split('\t', 1)[0] for i in l]
tag_names = [i.split('_landmarks.tag')[0] for i in tag_files]


mnc_names = [i.split('.mnc')[0] for i in mnc_files]

for i in mnc_names:
    tag_path = i+'_landmarks.tag'
    mnc_path = i+.mnc
