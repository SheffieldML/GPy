# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
The module contains the tools for ploting 2D image visualizations
"""

import numpy as np
from matplotlib.cm import jet

width_max = 15
height_max = 12

def _calculateFigureSize(x_size, y_size, fig_ncols, fig_nrows, pad):
    width = (x_size*fig_ncols+pad*(fig_ncols-1))
    height = (y_size*fig_nrows+pad*(fig_nrows-1))
    if width > float(height)/height_max*width_max:
        return (width_max, float(width_max)/width*height)
    else:
        return (float(height_max)/height*width, height_max)

def plot_2D_images(figure, arr, symmetric=False, pad=None, zoom=None, mode=None, interpolation='nearest'):
    ax = figure.add_subplot(111)
    if len(arr.shape)==2:
        arr = arr.reshape(*((1,)+arr.shape))
    fig_num = arr.shape[0]
    y_size = arr.shape[1]
    x_size = arr.shape[2]
    fig_ncols = int(np.ceil(np.sqrt(fig_num)))
    fig_nrows = int(np.ceil((float)(fig_num)/fig_ncols))
    if pad==None:
        pad = max(int(min(y_size,x_size)/10),1)
    
    figsize = _calculateFigureSize(x_size, y_size, fig_ncols, fig_nrows, pad)
    #figure.set_size_inches(figsize,forward=True)
    #figure.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    
    if symmetric:
        # symmetric around zero: fix zero as the middle color
        mval = max(abs(arr.max()),abs(arr.min()))
        arr = arr/(2.*mval)+0.5
    else:
        minval,maxval = arr.min(),arr.max()
        arr = (arr-minval)/(maxval-minval)

    if mode=='L':
        arr_color = np.empty(arr.shape+(3,))
        arr_color[:] = arr.reshape(*(arr.shape+(1,)))
    elif mode==None or mode=='jet':
        arr_color = jet(arr)
    
    buf = np.ones((y_size*fig_nrows+pad*(fig_nrows-1), x_size*fig_ncols+pad*(fig_ncols-1), 3),dtype=arr.dtype)
    
    for y in range(fig_nrows):
        for x in range(fig_ncols):
            if y*fig_ncols+x<fig_num:
                buf[y*y_size+y*pad:(y+1)*y_size+y*pad, x*x_size+x*pad:(x+1)*x_size+x*pad] = arr_color[y*fig_ncols+x,:,:,:3]
    img_plot = ax.imshow(buf, interpolation=interpolation)
    ax.axis('off')
