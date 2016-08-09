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
import GPy

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

    def test_offset_cluster(self):
        #Tests the GPy.util.cluster_with_offset.cluster utility with a small
        #test data set. Not using random noise just in case it occasionally
        #causes it not to cluster correctly.
        #groundtruth cluster identifiers are: [0,1,1,0]
        
        #data contains a list of the four sets of time series (3 per data point)      

        data = [np.array([[ 2.18094245,  1.96529789,  2.00265523,  2.18218742,  2.06795428],
                [ 1.62254829,  1.75748448,  1.83879347,  1.87531326,  1.52503496],
                [ 1.54589609,  1.61607914,  2.00463192,  1.48771394,  1.63339218]]),
         np.array([[ 2.86766106,  2.97953437,  2.91958876,  2.92510506,  3.03239241],
                [ 2.57368423,  2.59954886,  3.10000395,  2.75806125,  2.89865704],
                [ 2.58916318,  2.53698259,  2.63858411,  2.63102504,  2.51853901]]),
         np.array([[ 2.77834168,  2.9618564 ,  2.88482141,  3.24259745,  2.9716821 ],
                [ 2.60675576,  2.67095624,  2.94824436,  2.80520631,  2.87247516],
                [ 2.49543562,  2.5492281 ,  2.6505866 ,  2.65015308,  2.59738616]]),
         np.array([[ 1.76783086,  2.21666738,  2.07939706,  1.9268263 ,  2.23360121],
                [ 1.94305547,  1.94648592,  2.1278921 ,  2.09481457,  2.08575238],
                [ 1.69336013,  1.72285186,  1.6339506 ,  1.61212022,  1.39198698]])]

        #inputs contains their associated X values
        
        inputs = [np.array([[ 0.        ],
                [ 0.68040097],
                [ 1.20316795],
                [ 1.798749  ],
                [ 2.14891733]]), np.array([[ 0.        ],
                [ 0.51910637],
                [ 0.98259352],
                [ 1.57442965],
                [ 1.82515098]]), np.array([[ 0.        ],
                [ 0.66645478],
                [ 1.59464591],
                [ 1.69769551],
                [ 1.80932752]]), np.array([[ 0.        ],
                [ 0.87512108],
                [ 1.71881079],
                [ 2.67162871],
                [ 3.23761907]])]
                    
        #try doing the clustering
        active = GPy.util.cluster_with_offset.cluster(data,inputs)
        #check to see that the clustering has correctly clustered the time series.
        clusters = set([frozenset(cluster) for cluster in active])
        assert set([1,2]) in clusters, "Offset Clustering algorithm failed"
        assert set([0,3]) in clusters, "Offset Clustering algoirthm failed"
