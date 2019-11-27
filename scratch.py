import numpy as np
import ThreeDLabeler as td
import nibabel as nib 

im = nib.load('./MouseSkullData/475.mnc')
pt = td.tag_parser('MouseSkullData/475_landmarks.tag')

img = td.Image(im.get_data(), im.header.get_zooms(), pt)

img_flip0 = td.Image(np.flip(im.get_data(), 0),
                            im.header.get_zooms(),
                            np.flip(pt, 0))

img_flip1 = td.Image(np.flip(im.get_data(), 1),
                            im.header.get_zooms(),
                            np.flip(pt, 1))


# We need not rotate the tag file but instead switch columns or take 
# 128-column value