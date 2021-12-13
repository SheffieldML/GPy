"""
Created on 10 Sep 2012

@author: Max Zwiessele
@copyright: Max Zwiessele 2012
"""
import numpy

try:
    import pylab
    import matplotlib
except:
    pass
from operator import setitem
from functools import reduce


class PCA(object):
    """
    PCA module with automatic primal/dual determination.
    """

    def __init__(self, X):
        self.mu = None
        self.sigma = None

        X = self.center(X)

        # self.X = input
        if X.shape[0] >= X.shape[1]:
            # print "N >= D: using primal"
            self.eigvals, self.eigvectors = self._primal_eig(X)
        else:
            # print "N < D: using dual"
            self.eigvals, self.eigvectors = self._dual_eig(X)
        self.sort = numpy.argsort(self.eigvals)[::-1]
        self.eigvals = self.eigvals[self.sort]
        self.eigvectors = self.eigvectors[:, self.sort]
        self.fracs = self.eigvals / self.eigvals.sum()
        self.Q = self.eigvals.shape[0]

    def center(self, X):
        """
        Center `X` in PCA space.
        """
        X = X.copy()
        inan = numpy.isnan(X)
        if self.mu is None:
            X_ = numpy.ma.masked_array(X, inan)
            self.mu = X_.mean(0).base
            self.sigma = X_.std(0).base
        reduce(lambda y, x: setitem(x[0], x[1], x[2]), zip(X.T, inan.T, self.mu), None)
        X = X - self.mu
        X = X / numpy.where(self.sigma == 0, 1e-30, self.sigma)
        return X

    def _primal_eig(self, X):
        return numpy.linalg.eigh(numpy.einsum("ji,jk->ik", X, X))

    def _dual_eig(self, X):
        dual_eigvals, dual_eigvects = numpy.linalg.eigh(numpy.einsum("ij,kj->ik", X, X))
        relevant_dimensions = numpy.argsort(numpy.abs(dual_eigvals))[-X.shape[1] :]
        eigvals = dual_eigvals[relevant_dimensions]
        eigvects = dual_eigvects[:, relevant_dimensions]
        eigvects = (1.0 / numpy.sqrt(X.shape[0] * numpy.abs(eigvals))) * X.T.dot(
            eigvects
        )
        eigvects /= numpy.sqrt(numpy.diag(eigvects.T.dot(eigvects)))
        return eigvals, eigvects

    def project(self, X, Q=None):
        """
        Project X into PCA space, defined by the Q highest eigenvalues.
        Y = X dot V
        """
        if Q is None:
            Q = self.Q
        if Q > X.shape[1]:
            raise IndexError("requested dimension larger then input dimension")
        X = self.center(X)
        return X.dot(self.eigvectors[:, :Q])

    def plot_fracs(self, Q=None, ax=None, fignum=None):
        """
        Plot fractions of Eigenvalues sorted in descending order.
        """
        from ..plotting import Tango

        Tango.reset()
        col = Tango.nextMedium()
        if ax is None:
            fig = pylab.figure(fignum)
            ax = fig.add_subplot(111)
        if Q is None:
            Q = self.Q
        ticks = numpy.arange(Q)
        bar = ax.bar(ticks - 0.4, self.fracs[:Q], color=col)
        ax.set_xticks(ticks, map(lambda x: r"${}$".format(x), ticks + 1))
        ax.set_ylabel("Eigenvalue fraction")
        ax.set_xlabel("PC")
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlim(ticks.min() - 0.5, ticks.max() + 0.5)
        try:
            pylab.tight_layout()
        except:
            pass
        return bar

    def plot_2d(
        self,
        X,
        labels=None,
        s=20,
        marker="o",
        dimensions=(0, 1),
        ax=None,
        colors=None,
        fignum=None,
        cmap=None,  # @UndefinedVariable
        **kwargs
    ):
        """
        Plot dimensions `dimensions` with given labels against each other in
        PC space. Labels can be any sequence of labels of dimensions X.shape[0].
        Labels can be drawn with a subsequent call to legend()
        """
        if cmap is None:
            cmap = matplotlib.cm.jet
        if ax is None:
            fig = pylab.figure(fignum)
            ax = fig.add_subplot(111)
        if labels is None:
            labels = numpy.zeros(X.shape[0])
        ulabels = []
        for lab in labels:
            if lab not in ulabels:
                ulabels.append(lab)
        nlabels = len(ulabels)
        if colors is None:
            colors = iter([cmap(float(i) / nlabels) for i in range(nlabels)])
        else:
            colors = iter(colors)
        X_ = self.project(X, self.Q)[:, dimensions]
        kwargs.update(dict(s=s))
        plots = list()
        for i, l in enumerate(ulabels):
            kwargs.update(dict(color=next(colors), marker=marker[i % len(marker)]))
            plots.append(ax.scatter(*X_[labels == l, :].T, label=str(l), **kwargs))
        ax.set_xlabel(r"PC$_1$")
        ax.set_ylabel(r"PC$_2$")
        try:
            pylab.tight_layout()
        except:
            pass
        return plots
