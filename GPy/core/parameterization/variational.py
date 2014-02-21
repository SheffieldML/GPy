'''
Created on 6 Nov 2013

@author: maxz
'''
from parameterized import Parameterized
from param import Param
from transformations import Logexp

class Normal(Parameterized):
    '''
    Normal distribution for variational approximations.

    holds the means and variances for a factorizing multivariate normal distribution
    '''
    def __init__(self, means, variances, name='latent space'):
        Parameterized.__init__(self, name=name)
        self.mean = Param("mean", means)
        self.variance = Param('variance', variances, Logexp())
        self.add_parameters(self.mean, self.variance)

    def plot(self, *args):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot(self,*args)


class SpikeAndSlab(Parameterized):
    '''
    The SpikeAndSlab distribution for variational approximations.
    '''
    def __init__(self, means, variances, binary_prob, name='latent space'):
        """
        binary_prob : the probability of the distribution on the slab part.
        """
        Parameterized.__init__(self, name=name)
        self.mean = Param("mean", means)
        self.variance = Param('variance', variances, Logexp())
        self.gamma = Param("binary_prob",binary_prob,)
        self.add_parameters(self.mean, self.variance, self.gamma)

    def plot(self, *args):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot(self,*args)
