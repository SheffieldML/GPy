import pylab as pb
import numpy as np
from latent_space_visualizations.controllers.imshow_controller import ImshowController,ImAnnotateController
from ...util.misc import param_to_array
from ...core.parameterization.variational import VariationalPosterior
from .base_plots import x_frame2D
import itertools
import Tango
from matplotlib.cm import get_cmap

def most_significant_input_dimensions(model, which_indices):
    """
    Determine which dimensions should be plotted
    """
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
                plot_limits=None,
                aspect='auto', updates=False, predict_kwargs={}, imshow_kwargs={}):
    """
    :param labels: a np.array of size model.num_data containing labels for the points (can be number, strings, etc)
    :param resolution: the resolution of the grid on which to evaluate the predictive variance
    """
    if ax is None:
        fig = pb.figure(num=fignum)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    Tango.reset()

    if labels is None:
        labels = np.ones(model.num_data)

    input_1, input_2 = most_significant_input_dimensions(model, which_indices)

    #fethch the data points X that we'd like to plot
    X = model.X
    if isinstance(X, VariationalPosterior):
        X = param_to_array(X.mean)
    else:
        X = param_to_array(X)


    # create a function which computes the shading of latent space according to the output variance
    def plot_function(x):
        Xtest_full = np.zeros((x.shape[0], model.X.shape[1]))
        Xtest_full[:, [input_1, input_2]] = x
        _, var = model.predict(Xtest_full, **predict_kwargs)
        var = var[:, :1]
        return np.log(var)

    #Create an IMshow controller that can re-plot the latent space shading at a good resolution
    if plot_limits is None:
        xmin, ymin = X[:, [input_1, input_2]].min(0)
        xmax, ymax = X[:, [input_1, input_2]].max(0)
        x_r, y_r = xmax-xmin, ymax-ymin
        xmin -= .1*x_r
        xmax += .1*x_r
        ymin -= .1*y_r
        ymax += .1*y_r
    else:
        try:
            xmin, xmax, ymin, ymax = plot_limits
        except (TypeError, ValueError) as e:
            raise e.__class__, "Wrong plot limits: {} given -> need (xmin, xmax, ymin, ymax)".format(plot_limits)
    view = ImshowController(ax, plot_function,
                            (xmin, ymin, xmax, ymax),
                            resolution, aspect=aspect, interpolation='bilinear',
                            cmap=pb.cm.binary, **imshow_kwargs)

    # make sure labels are in order of input:
    labels = np.asarray(labels)
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
            this_label = unicode(ul)
        m = marker.next()

        index = np.nonzero(labels == ul)[0]
        if model.input_dim == 1:
            x = X[index, input_1]
            y = np.zeros(index.size)
        else:
            x = X[index, input_1]
            y = X[index, input_2]
        ax.scatter(x, y, marker=m, s=s, color=Tango.nextMedium(), label=this_label)

    ax.set_xlabel('latent dimension %i' % input_1)
    ax.set_ylabel('latent dimension %i' % input_2)

    if not np.all(labels == 1.) and legend:
        ax.legend(loc=0, numpoints=1)

    ax.grid(b=False) # remove the grid if present, it doesn't look good
    ax.set_aspect('auto') # set a nice aspect ratio

    if plot_inducing:
        Z = param_to_array(model.Z)
        ax.plot(Z[:, input_1], Z[:, input_2], '^w')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    try:
        fig.canvas.draw()
        fig.tight_layout()
        fig.canvas.draw()
    except Exception as e:
        print "Could not invoke tight layout: {}".format(e)
        pass

    if updates:
        try:
            ax.figure.canvas.show()
        except Exception as e:
            print "Could not invoke show: {}".format(e)
        raw_input('Enter to continue')
        view.deactivate()
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
    Tango.reset()

    if labels is None:
        labels = np.ones(model.num_data)

    input_1, input_2 = most_significant_input_dimensions(model, which_indices)

    # first, plot the output variance as a function of the latent space
    Xtest, xx, yy, xmin, xmax = x_frame2D(model.X[:, [input_1, input_2]], resolution=resolution)
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
        ax.scatter(x, y, marker=m, s=s, color=Tango.nextMedium(), label=this_label)

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
        fig.canvas.show()
        raw_input('Enter to continue')

    pb.title('Magnification Factor')
    return ax


def plot_steepest_gradient_map(model, fignum=None, ax=None, which_indices=None, labels=None, data_labels=None, data_marker='o', data_s=40, resolution=20, aspect='auto', updates=False, ** kwargs):

    input_1, input_2 = significant_dims = most_significant_input_dimensions(model, which_indices)

    X = np.zeros((resolution ** 2, model.input_dim))
    indices = np.r_[:X.shape[0]]
    if labels is None:
        labels = range(model.output_dim)

    def plot_function(x):
        X[:, significant_dims] = x
        dmu_dX = model.dmu_dXnew(X)
        argmax = np.argmax(dmu_dX, 1)
        return dmu_dX[indices, argmax], np.array(labels)[argmax]

    if ax is None:
        fig = pb.figure(num=fignum)
        ax = fig.add_subplot(111)

    if data_labels is None:
        data_labels = np.ones(model.num_data)
    ulabels = []
    for lab in data_labels:
        if not lab in ulabels:
            ulabels.append(lab)
    marker = itertools.cycle(list(data_marker))
    for i, ul in enumerate(ulabels):
        if type(ul) is np.string_:
            this_label = ul
        elif type(ul) is np.int64:
            this_label = 'class %i' % ul
        else:
            this_label = 'class %i' % i
        m = marker.next()
        index = np.nonzero(data_labels == ul)[0]
        x = model.X[index, input_1]
        y = model.X[index, input_2]
        ax.scatter(x, y, marker=m, s=data_s, color=Tango.nextMedium(), label=this_label)

    ax.set_xlabel('latent dimension %i' % input_1)
    ax.set_ylabel('latent dimension %i' % input_2)

    controller = ImAnnotateController(ax,
                                  plot_function,
                                  tuple(model.X.min(0)[:, significant_dims]) + tuple(model.X.max(0)[:, significant_dims]),
                                  resolution=resolution,
                                  aspect=aspect,
                                  cmap=get_cmap('jet'),
                                  **kwargs)
    ax.legend()
    ax.figure.tight_layout()
    if updates:
        pb.show()
        clear = raw_input('Enter to continue')
        if clear.lower() in 'yes' or clear == '':
            controller.deactivate()
    return controller.view
