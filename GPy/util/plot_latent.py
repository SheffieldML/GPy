import pylab as pb
import numpy as np
from .. import util
from .latent_space_visualizations.controllers.imshow_controller import ImshowController
import itertools

def most_significant_input_dimensions(model, which_indices):
    if which_indices is None:
        if model.input_dim == 1:
            input_1 = 0
            input_2 = None
        if model.input_dim == 2:
            input_1, input_2 = 0, 1
        else:
            try:
                input_1, input_2 = np.argsort(model.input_sensitivity())[::-1][:2]
            except:
                raise ValueError, "cannot automatically determine which dimensions to plot, please pass 'which_indices'"
    else:
        input_1, input_2 = which_indices
    return input_1, input_2

def plot_latent(model, labels=None, which_indices=None,
                resolution=50, ax=None, marker='o', s=40,
                fignum=None, plot_inducing=False, legend=True,
                aspect='auto', updates=False):
    """
    :param labels: a np.array of size model.num_data containing labels for the points (can be number, strings, etc)
    :param resolution: the resolution of the grid on which to evaluate the predictive variance
    """
    if ax is None:
        fig = pb.figure(num=fignum)
        ax = fig.add_subplot(111)
    util.plot.Tango.reset()

    if labels is None:
        labels = np.ones(model.num_data)

    input_1, input_2 = most_significant_input_dimensions(model, which_indices)

    # first, plot the output variance as a function of the latent space
    Xtest, xx, yy, xmin, xmax = util.plot.x_frame2D(model.X[:, [input_1, input_2]], resolution=resolution)
    #Xtest_full = np.zeros((Xtest.shape[0], model.X.shape[1]))
    Xtest_full = np.zeros((Xtest.shape[0], model.X.shape[1]))
    def plot_function(x):
        Xtest_full[:, [input_1, input_2]] = x
        mu, var, low, up = model.predict(Xtest_full)
        var = var[:, :1]
        return np.log(var)
    
    xmi, ymi = xmin
    xma, yma = xmax
    
    view = ImshowController(ax, plot_function,
                            (xmi, ymi, xma, yma),
                            resolution, aspect=aspect, interpolation='bilinear',
                            cmap=pb.cm.binary)

#     ax.imshow(var.reshape(resolution, resolution).T,
#               extent=[xmin[0], xmax[0], xmin[1], xmax[1]], cmap=pb.cm.binary, interpolation='bilinear', origin='lower')

    # make sure labels are in order of input:
    ulabels = []
    for lab in labels:
        if not lab in ulabels:
            ulabels.append(lab)

    marker = itertools.cycle(list(marker))

    for i, ul in enumerate(ulabels):
        if type(ul) is np.string_:
            this_label = ul
        elif type(ul) is np.int64:
            this_label = 'class %i' % ul
        else:
            this_label = 'class %i' % i
        m = marker.next()

        index = np.nonzero(labels == ul)[0]
        if model.input_dim == 1:
            x = model.X[index, input_1]
            y = np.zeros(index.size)
        else:
            x = model.X[index, input_1]
            y = model.X[index, input_2]
        ax.scatter(x, y, marker=m, s=s, color=util.plot.Tango.nextMedium(), label=this_label)

    ax.set_xlabel('latent dimension %i' % input_1)
    ax.set_ylabel('latent dimension %i' % input_2)

    if not np.all(labels == 1.) and legend:
        ax.legend(loc=0, numpoints=1)

    ax.set_xlim(xmin[0], xmax[0])
    ax.set_ylim(xmin[1], xmax[1])
    ax.grid(b=False) # remove the grid if present, it doesn't look good
    ax.set_aspect('auto') # set a nice aspect ratio

    if plot_inducing:
        ax.plot(model.Z[:, input_1], model.Z[:, input_2], '^w')

    if updates:
        ax.figure.canvas.show()
        raw_input('Enter to continue')
    return ax

def plot_magnification(model, labels=None, which_indices=None,
                resolution=60, ax=None, marker='o', s=40,
                fignum=None, plot_inducing=False, legend=True,
                aspect='auto', updates=False):
    """
    :param labels: a np.array of size model.num_data containing labels for the points (can be number, strings, etc)
    :param resolution: the resolution of the grid on which to evaluate the predictive variance
    """
    if ax is None:
        fig = pb.figure(num=fignum)
        ax = fig.add_subplot(111)
    util.plot.Tango.reset()

    if labels is None:
        labels = np.ones(model.num_data)

    input_1, input_2 = most_significant_input_dimensions(model, which_indices)

    # first, plot the output variance as a function of the latent space
    Xtest, xx, yy, xmin, xmax = util.plot.x_frame2D(model.X[:, [input_1, input_2]], resolution=resolution)
    Xtest_full = np.zeros((Xtest.shape[0], model.X.shape[1]))
    def plot_function(x):
        Xtest_full[:, [input_1, input_2]] = x
        mf=model.magnification(Xtest_full)
        return mf
    view = ImshowController(ax, plot_function,
                            tuple(model.X.min(0)[:, [input_1, input_2]]) + tuple(model.X.max(0)[:, [input_1, input_2]]),
                            resolution, aspect=aspect, interpolation='bilinear',
                            cmap=pb.cm.gray)

    # make sure labels are in order of input:
    ulabels = []
    for lab in labels:
        if not lab in ulabels:
            ulabels.append(lab)

    marker = itertools.cycle(list(marker))

    for i, ul in enumerate(ulabels):
        if type(ul) is np.string_:
            this_label = ul
        elif type(ul) is np.int64:
            this_label = 'class %i' % ul
        else:
            this_label = 'class %i' % i
        m = marker.next()

        index = np.nonzero(labels == ul)[0]
        if model.input_dim == 1:
            x = model.X[index, input_1]
            y = np.zeros(index.size)
        else:
            x = model.X[index, input_1]
            y = model.X[index, input_2]
        ax.scatter(x, y, marker=m, s=s, color=util.plot.Tango.nextMedium(), label=this_label)

    ax.set_xlabel('latent dimension %i' % input_1)
    ax.set_ylabel('latent dimension %i' % input_2)

    if not np.all(labels == 1.) and legend:
        ax.legend(loc=0, numpoints=1)

    ax.set_xlim(xmin[0], xmax[0])
    ax.set_ylim(xmin[1], xmax[1])
    ax.grid(b=False) # remove the grid if present, it doesn't look good
    ax.set_aspect('auto') # set a nice aspect ratio

    if plot_inducing:
        ax.plot(model.Z[:, input_1], model.Z[:, input_2], '^w')

    if updates:
        ax.figure.canvas.show()
        raw_input('Enter to continue')

    pb.title('Magnification Factor')
    return ax
