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
        canvas, kwargs = pl.get_new_canvas(**kwargs)
        plots = dict(trace=pl.plot(range(len(optimizer.trace)), optimizer.trace))
        return pl.show_canvas(canvas, plots, xlabel='Iteration', ylabel='f(x)')

def plot_sgd_traces(optimizer):
    pb.figure()
    pb.subplot(211)
    pb.title('Parameters')
    for k in optimizer.param_traces.keys():
        pb.plot(optimizer.param_traces[k], label=k)
    pb.legend(loc=0)
    pb.subplot(212)
    pb.title('Objective function')
    pb.plot(optimizer.fopt_trace)
