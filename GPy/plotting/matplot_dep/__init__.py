# Copyright (c) 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from matplotlib import pyplot as plt
from . import defaults

def get_new_canvas(kwargs):
    """
    Return a canvas, kwargupdate for matplotlib. This just a 
    dictionary for the collection and we add the an axis to kwarg.
    
    This method does two things, it creates an empty canvas
    and updates the kwargs (deletes the unnecessary kwargs)
    for further usage in normal plotting.
    
    in matplotlib this means it deletes references to ax, as
    plotting is done on the axis itself and is not a kwarg. 
    """
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    elif 'num' in kwargs and 'figsize' in kwargs:
        ax = plt.figure(num=kwargs.pop('num'), figsize=kwargs.pop('figsize')).add_subplot(111) 
    elif 'num' in kwargs:
        ax = plt.figure(num=kwargs.pop('num')).add_subplot(111)
    elif 'figsize' in kwargs:
        ax = plt.figure(figsize=kwargs.pop('figsize')).add_subplot(111)
    else:
        ax = plt.figure().add_subplot(111)
    # Add ax to kwargs to add all subsequent plots to this axis:
    #kwargs['ax'] = ax
    return ax, kwargs

def show_canvas(canvas):
    try:
        canvas.figure.canvas.draw()
        canvas.figure.tight_layout()
    except:
        pass
    return canvas


def scatter(ax, *args, **kwargs):
    ax.scatter(*args, **kwargs)

def plot(ax, *args, **kwargs):
    ax.plot(*args, **kwargs)

def imshow(ax, *args, **kwargs):
    ax.imshow(*args, **kwargs)

    
from . import base_plots
from . import models_plots
from . import priors_plots
from . import variational_plots
from . import kernel_plots
from . import dim_reduction_plots
from . import mapping_plots
from . import Tango
from . import visualize
from . import latent_space_visualizations
from . import inference_plots
from . import maps
from . import img_plots
from .ssgplvm import SSGPLVM_plot


