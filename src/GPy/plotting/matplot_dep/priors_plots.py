# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
try:
    from matplotlib import pyplot as pb
except:
    pass


def univariate_plot(prior):
    rvs = prior.rvs(1000)
    pb.hist(rvs, 100, normed=True)
    xmin, xmax = pb.xlim()
    xx = np.linspace(xmin, xmax, 1000)
    pb.plot(xx, prior.pdf(xx), 'r', linewidth=2)

def plot(prior):

    if prior.input_dim == 2:
        rvs = prior.rvs(200)
        pb.plot(rvs[:, 0], rvs[:, 1], 'kx', mew=1.5)
        xmin, xmax = pb.xlim()
        ymin, ymax = pb.ylim()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        xflat = np.vstack((xx.flatten(), yy.flatten())).T
        zz = prior.pdf(xflat).reshape(100, 100)
        pb.contour(xx, yy, zz, linewidths=2)

    else:
        raise NotImplementedError("Cannot define a frame with more than two input dimensions")
