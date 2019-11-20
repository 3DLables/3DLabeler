import matplotlib.pyplot as plt
import numpy as np

def mri_plot(img, points, vcol=1):
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
            im = img[:, :, im_slice]
        elif vcol == 1:
            im = img[:, im_slice, :]
        else:
            im = img[im_slice, :, :]
        ax.append( fig.add_subplot(rows, columns, i+1))
        ax[-1].set_title("Image depth: "+str(im_slice))  # set title
        plt.imshow(im)
        plot_cols = np.array([0, 1, 2])
        plot_cols = plot_cols[plot_cols != vcol]
        plt.plot(points[i, min(plot_cols)], points[i, max(plot_cols)], 'ro')

    plt.show()


