# Copyright (c) 2012, James Hensman and Nicolo' Fusi
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from matplotlib import pyplot as pb


def plot(model, ax=None, fignum=None, Z_height=None, **kwargs):

    if ax is None:
        fig = pb.figure(num=fignum)
        ax = fig.add_subplot(111)

    #horrible hack here:
    data = model.likelihood.data.copy()
    model.likelihood.data = model.Y
    GP.plot(model, ax=ax, **kwargs)
    model.likelihood.data = data

    Zu = model.Z * model._Xscale + model._Xoffset
    if model.input_dim==1:
        ax.plot(model.X_batch, model.likelihood.data, 'gx',mew=2)
        if Z_height is None:
            Z_height = ax.get_ylim()[0]
        ax.plot(Zu, np.zeros_like(Zu) + Z_height, 'r|', mew=1.5, markersize=12)

    if model.input_dim==2:
        ax.scatter(model.X[:,0], model.X[:,1], 20., model.Y[:,0], linewidth=0, cmap=pb.cm.jet)  # @UndefinedVariable
        ax.plot(Zu[:,0], Zu[:,1], 'w^')

def plot_traces(model):

    pb.figure()
    t = np.array(model._param_trace)
    pb.subplot(2,1,1)
    for l,ti in zip(model._get_param_names(),t.T):
        if not l[:3]=='iip':
            pb.plot(ti,label=l)
    pb.legend(loc=0)

    pb.subplot(2,1,2)
    pb.plot(np.asarray(model._ll_trace),label='stochastic likelihood')
    pb.legend(loc=0)
