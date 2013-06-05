import pylab as pb
import numpy as np
from .. import util

def plot_latent(model, labels=None, which_indices=None, resolution=50, ax=None, marker='o', s=40):
    """
    :param labels: a np.array of size model.num_data containing labels for the points (can be number, strings, etc)
    :param resolution: the resolution of the grid on which to evaluate the predictive variance
    """
    if ax is None:
        ax = pb.gca()
    util.plot.Tango.reset()

    if labels is None:
        labels = np.ones(model.num_data)
    if which_indices is None:
        if model.input_dim==1:
            input_1 = 0
            input_2 = None
        if model.input_dim==2:
            input_1, input_2 = 0,1
        else:
            try:
                input_1, input_2 = np.argsort(model.input_sensitivity())[:2]
            except:
                raise ValueError, "cannot Atomatically determine which dimensions to plot, please pass 'which_indices'"
    else:
        input_1, input_2 = which_indices

    #first, plot the output variance as a function of the latent space
    Xtest, xx,yy,xmin,xmax = util.plot.x_frame2D(model.X[:,[input_1, input_2]],resolution=resolution)
    Xtest_full = np.zeros((Xtest.shape[0], model.X.shape[1]))
    Xtest_full[:, :2] = Xtest
    mu, var, low, up = model.predict(Xtest_full)
    var = var[:, :1]
    ax.imshow(var.reshape(resolution, resolution).T,
              extent=[xmin[0], xmax[0], xmin[1], xmax[1]], cmap=pb.cm.binary,interpolation='bilinear',origin='lower')

    # make sure labels are in order of input:
    ulabels = []
    for lab in labels:
        if not lab in ulabels:
            ulabels.append(lab)

    for i, ul in enumerate(ulabels):
        if type(ul) is np.string_:
            this_label = ul
        elif type(ul) is np.int64:
            this_label = 'class %i'%ul
        else:
            this_label = 'class %i'%i
        if len(marker) == len(ulabels):
            m = marker[i]
        else:
            m = marker

        index = np.nonzero(labels==ul)[0]
        if model.input_dim==1:
            x = model.X[index,input_1]
            y = np.zeros(index.size)
        else:
            x = model.X[index,input_1]
            y = model.X[index,input_2]
        ax.scatter(x, y, marker=m, s=s, color=util.plot.Tango.nextMedium(), label=this_label)

    ax.set_xlabel('latent dimension %i'%input_1)
    ax.set_ylabel('latent dimension %i'%input_2)

    if not np.all(labels==1.):
        ax.legend(loc=0,numpoints=1)

    ax.set_xlim(xmin[0],xmax[0])
    ax.set_ylim(xmin[1],xmax[1])
    ax.grid(b=False) # remove the grid if present, it doesn't look good
    ax.set_aspect('auto') # set a nice aspect ratio
    return ax


def plot_latent_indices(Model, which_indices=None, *args, **kwargs):

    if which_indices is None:
        try:
            input_1, input_2 = np.argsort(Model.input_sensitivity())[:2]
        except:
            raise ValueError, "cannot Automatically determine which dimensions to plot, please pass 'which_indices'"
    else:
        input_1, input_2 = which_indices
    ax = plot_latent(Model, which_indices=[input_1, input_2], *args, **kwargs)
    # TODO: Here test if there are inducing points...
    ax.plot(Model.Z[:, input_1], Model.Z[:, input_2], '^w')
    return ax
