#===============================================================================
# Copyright (c) 2016, Max Zwiessele, Alan Saul
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of GPy.testing.util_tests nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#===============================================================================

import unittest, numpy as np

class TestDebug(unittest.TestCase):
    def test_checkFinite(self):
        from GPy.util.debug import checkFinite
        array = np.random.normal(0, 1, 100).reshape(25,4)
        self.assertTrue(checkFinite(array, name='test'))
        
        array[np.random.binomial(1, .3, array.shape).astype(bool)] = np.nan
        self.assertFalse(checkFinite(array))

    def test_checkFullRank(self):
        from GPy.util.debug import checkFullRank
        from GPy.util.linalg import tdot
        array = np.random.normal(0, 1, 100).reshape(25,4)
        self.assertFalse(checkFullRank(tdot(array), name='test'))
        
        array = np.random.normal(0, 1, (25,25))
        self.assertTrue(checkFullRank(tdot(array)))
    
    def test_fixed_inputs_median(self):
        """ test fixed_inputs convenience function """
        from GPy.plotting.matplot_dep.util import fixed_inputs
        import GPy
        X = np.random.randn(10, 3)
        Y = np.sin(X) + np.random.randn(10, 3)*1e-3
        m = GPy.models.GPRegression(X, Y)
        fixed = fixed_inputs(m, [1], fix_routine='median', as_list=True, X_all=False)
        self.assertTrue((0, np.median(X[:,0])) in fixed)
        self.assertTrue((2, np.median(X[:,2])) in fixed)
        self.assertTrue(len([t for t in fixed if t[0] == 1]) == 0) # Unfixed input should not be in fixed

    def test_fixed_inputs_mean(self):
        from GPy.plotting.matplot_dep.util import fixed_inputs
        import GPy
        X = np.random.randn(10, 3)
        Y = np.sin(X) + np.random.randn(10, 3)*1e-3
        m = GPy.models.GPRegression(X, Y)
        fixed = fixed_inputs(m, [1], fix_routine='mean', as_list=True, X_all=False)
        self.assertTrue((0, np.mean(X[:,0])) in fixed)
        self.assertTrue((2, np.mean(X[:,2])) in fixed)
        self.assertTrue(len([t for t in fixed if t[0] == 1]) == 0) # Unfixed input should not be in fixed

    def test_fixed_inputs_zero(self):
        from GPy.plotting.matplot_dep.util import fixed_inputs
        import GPy
        X = np.random.randn(10, 3)
        Y = np.sin(X) + np.random.randn(10, 3)*1e-3
        m = GPy.models.GPRegression(X, Y)
        fixed = fixed_inputs(m, [1], fix_routine='zero', as_list=True, X_all=False)
        self.assertTrue((0, 0.0) in fixed)
        self.assertTrue((2, 0.0) in fixed)
        self.assertTrue(len([t for t in fixed if t[0] == 1]) == 0) # Unfixed input should not be in fixed

    def test_fixed_inputs_uncertain(self):
        from GPy.plotting.matplot_dep.util import fixed_inputs
        import GPy
        from GPy.core.parameterization.variational import NormalPosterior
        X_mu = np.random.randn(10, 3)
        X_var = np.random.randn(10, 3)
        X = NormalPosterior(X_mu, X_var)
        Y = np.sin(X_mu) + np.random.randn(10, 3)*1e-3
        m = GPy.models.BayesianGPLVM(Y, X=X_mu, X_variance=X_var, input_dim=3)
        fixed = fixed_inputs(m, [1], fix_routine='median', as_list=True, X_all=False)
        self.assertTrue((0, np.median(X.mean.values[:,0])) in fixed)
        self.assertTrue((2, np.median(X.mean.values[:,2])) in fixed)
        self.assertTrue(len([t for t in fixed if t[0] == 1]) == 0) # Unfixed input should not be in fixed

    def test_subarray(self):
        import GPy
        X = np.zeros((3,6), dtype=bool)
        X[[1,1,1],[0,4,5]] = 1
        X[1:,[2,3]] = 1
        d = GPy.util.subarray_and_sorting.common_subarrays(X,axis=1)
        self.assertTrue(len(d) == 3)
        X[:, d[tuple(X[:,0])]]
        self.assertTrue(d[tuple(X[:,4])] == d[tuple(X[:,0])] == [0, 4, 5])
        self.assertTrue(d[tuple(X[:,1])] == [1])

