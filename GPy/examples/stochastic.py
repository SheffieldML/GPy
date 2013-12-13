# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import pylab as pb
import numpy as np
import GPy

def toy_1d(optimize=True, plot=True):
    N = 2000
    M = 20

    #create data
    X = np.linspace(0,32,N)[:,None]
    Z = np.linspace(0,32,M)[:,None]
    Y = np.sin(X) + np.cos(0.3*X) + np.random.randn(*X.shape)/np.sqrt(50.)

    m = GPy.models.SVIGPRegression(X,Y, batchsize=10, Z=Z)
    m.constrain_bounded('noise_variance',1e-3,1e-1)
    m.constrain_bounded('white_variance',1e-3,1e-1)

    m.param_steplength = 1e-4

    if plot:
        fig = pb.figure()
        ax = fig.add_subplot(111)
        def cb(foo):
            ax.cla()
            m.plot(ax=ax,Z_height=-3)
            ax.set_ylim(-3,3)
            fig.canvas.draw()

    if optimize:
        m.optimize(500, callback=cb, callback_interval=1)

    if plot:
        m.plot_traces()
    return m
