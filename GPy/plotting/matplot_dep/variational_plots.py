import pylab as pb, numpy as np
from ...util.misc import param_to_array

def plot(parameterized, fignum=None, ax=None, colors=None):
    """
    Plot latent space X in 1D:

        - if fig is given, create input_dim subplots in fig and plot in these
        - if ax is given plot input_dim 1D latent space plots of X into each `axis`
        - if neither fig nor ax is given create a figure with fignum and plot in there

    colors:
        colors of different latent space dimensions input_dim

    """
    if ax is None:
        fig = pb.figure(num=fignum, figsize=(8, min(12, (2 * parameterized.means.shape[1]))))
    if colors is None:
        colors = pb.gca()._get_lines.color_cycle
        pb.clf()
    else:
        colors = iter(colors)
    plots = []
    means, variances = param_to_array(parameterized.means, parameterized.variances)
    x = np.arange(means.shape[0])
    for i in range(means.shape[1]):
        if ax is None:
            a = fig.add_subplot(means.shape[1], 1, i + 1)
        elif isinstance(ax, (tuple, list)):
            a = ax[i]
        else:
            raise ValueError("Need one ax per latent dimension input_dim")
        a.plot(means, c='k', alpha=.3)
        plots.extend(a.plot(x, means.T[i], c=colors.next(), label=r"$\mathbf{{X_{{{}}}}}$".format(i)))
        a.fill_between(x,
                        means.T[i] - 2 * np.sqrt(variances.T[i]),
                        means.T[i] + 2 * np.sqrt(variances.T[i]),
                        facecolor=plots[-1].get_color(),
                        alpha=.3)
        a.legend(borderaxespad=0.)
        a.set_xlim(x.min(), x.max())
        if i < means.shape[1] - 1:
            a.set_xticklabels('')
    pb.draw()
    fig.tight_layout(h_pad=.01) # , rect=(0, 0, 1, .95))
    return fig
