"""
The module plotting results for SSGPLVM
"""

from matplotlib import pyplot as plt

from ...models import SSGPLVM
from .img_plots import plot_2D_images


class SSGPLVM_plot:
    def __init__(self, model, imgsize):
        if not isinstance(model, SSGPLVM):
            raise TypeError(f"expected SSGPLVM, got {type(model).__name__}")
        self.model = model
        self.imgsize = imgsize
        if model.Y.shape[1] != imgsize[0] * imgsize[1]:
            raise ValueError("imgsize must match model.Y width")

    def plot_inducing(self):
        fig1 = plt.figure()
        mean = self.model.posterior.mean
        arr = mean.reshape(*(mean.shape[0], self.imgsize[1], self.imgsize[0]))
        plot_2D_images(fig1, arr)
        fig1.gca().set_title("The mean of inducing points")

        fig2 = plt.figure()
        covar = self.model.posterior.covariance
        plot_2D_images(fig2, covar)
        fig2.gca().set_title("The variance of inducing points")
