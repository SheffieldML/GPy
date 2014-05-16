# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ....core.parameterization.parameter_core import Pickleable

class PSICOMP(Pickleable):

    def psicomputations(self, variance, Z, variational_posterior):
        """
        Compute psi-statistics
        """
        pass
    
    def psiDerivativecomputations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, variance, Z, variational_posterior):
        """
        Compute the derivatives of parameters by combing dL_dpsi and dpsi_dparam
        """
        pass

