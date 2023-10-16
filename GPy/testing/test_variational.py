"""
Copyright (c) 2015, Max Zwiessele
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of paramax nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import GPy, numpy as np


class KLGrad(GPy.core.Model):
    def __init__(self, Xvar, kl):
        super(KLGrad, self).__init__(name="klgrad")
        self.kl = kl
        self.link_parameter(Xvar)
        self.Xvar = Xvar
        self._obj = 0

    def parameters_changed(self):
        self.Xvar.gradient[:] = 0
        self.kl.update_gradients_KL(self.Xvar)
        self._obj = self.kl.KL_divergence(self.Xvar)

    def objective_function(self):
        return self._obj


class TestVariational:
    def setup(self):
        np.random.seed(12345)
        self.Xvar = GPy.core.parameterization.variational.NormalPosterior(
            np.random.uniform(0, 1, (10, 3)), np.random.uniform(1e-5, 0.01, (10, 3))
        )

    def test_normal(self):
        self.setup()
        klgrad = KLGrad(self.Xvar, GPy.core.parameterization.variational.NormalPrior())
        np.testing.assert_(klgrad.checkgrad())
