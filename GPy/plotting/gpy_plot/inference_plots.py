# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

#import numpy as np
#import Tango
#from base_plots import gpplot, x_frame1D, x_frame2D

from . import plotting_library as pl

def plot_optimizer(optimizer, **kwargs):
    if optimizer.trace == None:
        print("No trace present so I can't plot it. Please check that the optimizer actually supplies a trace.")
    else:
        canvas, kwargs = pl().new_canvas(**kwargs)
        plots = dict(trace=pl().plot(range(len(optimizer.trace)), optimizer.trace))
        return pl().add_to_canvas(canvas, plots, xlabel='Iteration', ylabel='f(x)')

def plot_sgd_traces(optimizer):
    figure = pl().figure(2,1)
    canvas, _ = pl().new_canvas(figure, 1, 1, title="Parameters")
    plots = dict(lines=[])
    for k in optimizer.param_traces.keys():
        plots['lines'].append(pl().plot(canvas, range(len(optimizer.param_traces[k])), optimizer.param_traces[k], label=k))
    pl().add_to_canvas(canvas, legend=True)
    canvas, _ = pl().new_canvas(figure, 1, 2, title="Objective function")
    pl().plot(canvas, range(len(optimizer.fopt_trace)), optimizer.fopt_trace)
    return pl().add_to_canvas(canvas, plots, legend=True)
