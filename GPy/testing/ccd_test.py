import numpy as np
import matplotlib.pyplot as plt
import GPy
%matplotlib inline
np.set_printoptions(suppress=True, precision=10)
from paramz.transformations import Logexp

class SimpleModel(GPy.Model):
    def __init__(self, name):
        super(SimpleModel, self).__init__(name)
        self.params = []
        self.peak = np.array([1,2])
        for i,pos in enumerate(self.peak):
            p = GPy.Param('param%d' % i, 1.0)
            self.params.append(p)
            self.link_parameter(p)
    
    def log_likelihood(self):
        like = 0
        for i,pos in enumerate(self.peak):
            like -= ((self.params[i])-pos)**2
        return like
    
    def parameters_changed(self):
        for i,pos in enumerate(self.peak):
            self.params[i].gradient = -2*((self.params[i])-pos)
    
m2 = SimpleModel('simple')
m2.optimize()
assert np.all(np.isclose(m2.numerical_parameter_hessian(),np.array([[2,0],[0,2]]))), "Numerical approximation to Hessian doesn't match. Error in numerical_parameter_hessian()."
assert np.isclose(m2.param0,1,atol=0.01), "Failed to find likelihood maximum of test model's param0 while testing CCD"
assert np.isclose(m2.param1,2,atol=0.01), "Failed to find likelihood maximum of test model's param1 while testing CCD"

ccdpos,ccdres = m2.CCD()
m2.optimize()

assert np.all(np.isclose(np.sum((ccdpos-np.array([1,2]))**2,1)[1:],1.21,atol=0.01)), "CCD placement error - should be 1.21 from mode"
s
