'''
Created on 6 Nov 2013

@author: maxz
'''
from parameterized import Parameterized
from parameter import Param

class Normal(Parameterized):
    '''
    Normal distribution for variational approximations.
    
    holds the means and variances for a factorizing multivariate normal distribution
    '''
    def __init__(self, name, means, variances):
        Parameterized.__init__(self, name=name)
        self.means = Param("mean", means)
        self.variances = Param('variance', variances)
        self.add_parameters(self.means, self.variances)