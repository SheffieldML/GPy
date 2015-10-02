def update_not_existing_kwargs(to_update, update_from):
    return to_update.update({k:v for k,v in update_from.items() if k not in to_update})

#===============================================================================
# Implement library specific defaults in the specific plotting librarys defaults.py file.
# The following lines ensure, that an empty kwarg gets returned, when accessing a not 
# existing default
from .. import plotting_library as pl
from collections import defaultdict
class defaultdict(defaultdict):
    def __getattr__(self, *args, **kwargs):
        return defaultdict.__getitem__(self, *args, **kwargs)
defaults = defaultdict(dict, **pl.defaults.__dict__)
pl.defaults = defaults
#===============================================================================

#===============================================================================
# Make sure that the necessary files and functions are
# defined in the plotting library:
assert hasattr(pl, 'get_new_canvas'), "Please implement a function to get a new canvas for the specific library in plotting_library.get_new_canvas(**kwargs)"
assert hasattr(pl, 'plot'), "Please implement a function to plot a simple line"
assert hasattr(pl, 'scatter'), "Please implement a function to plot a simple scatterplot"
#assert hasattr(pl, 'xerrorbar'), "Please implement a function to plot an errorbar along the xaxis"
#assert hasattr(pl, 'xerrorbar'), "Please implement a function to plot an errorbar along the yaxis"
#assert hasattr(pl, 'fill'), "Please implement a function to fill a section between points"
#assert hasattr(pl, 'imshow'), "Please implement a function to plot an image in the given boundaries"
#===============================================================================

from . import data_plots, gp_plots
