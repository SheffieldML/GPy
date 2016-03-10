"""
The module plotting results for SSGPLVM
"""

import pylab

from ...models import SSGPLVM
from .img_plots import plot_2D_images

class SSGPLVM_plot(object):
    def __init__(self,model, imgsize):
        assert isinstance(model,SSGPLVM)
        self.model = model
        self.imgsize= imgsize
        assert model.Y.shape[1] == imgsize[0]*imgsize[1]

    def plot_inducing(self):
        fig1 = pylab.figure()
        mean = self.model.posterior.mean
        arr = mean.reshape(*(mean.shape[0],self.imgsize[1],self.imgsize[0]))
        plot_2D_images(fig1, arr)
        fig1.gca().set_title('The mean of inducing points')

        fig2 = pylab.figure()
        covar = self.model.posterior.covariance
        plot_2D_images(fig2, covar)
        fig2.gca().set_title('The variance of inducing points')

