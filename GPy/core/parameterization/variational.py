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
        self.means = Param("mean", means)
        self.variances = Param('variance', variances, Logexp())
        self.add_parameters(self.means, self.variances)

    def plot(self, *args):
        """
        Plot latent space X in 1D:

        See  GPy.plotting.matplot_dep.variational_plots
        """
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import variational_plots
        return variational_plots.plot(self,*args)
