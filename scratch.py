#%%
import nibabel as nib
import ThreeDLabeler as td 
# %%

im_nib = nib.load('MouseSkullData/475.mnc')

# %%
points = td.tag_parser('MouseSkullData/475_landmarks.tag')

# %%
img = td.Image(im_nib.get_data(), im_nib.header.get_zooms(), points)

# %%
img_test = img.cube()

# %%
img_scaled = img_test.scale()

# %%
img32 = img_scaled.scale(32)

# %%
img64 = img_scaled.scale(64)


# %%
