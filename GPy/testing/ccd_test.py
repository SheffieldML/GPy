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

class CCDTest(unittest.TestCase):
    def test_ccd_placement(self):
        """
        Test if the CCD algorithm placed the function evaluations at the
        expected locations in parameter space, for the simple model.
        """
        
        #create and optimise the simple model
        m2 = SimpleModel('simple')
        m2.optimize()
        #confirm that the gradients of the likelihood over the parameters
        #is as expected: [[2,0],[0,2]].
        assert np.all(np.isclose(m2.numerical_parameter_hessian(),np.array([[2,0],[0,2]]))), "Numerical approximation to Hessian doesn't match. Error in numerical_parameter_hessian()."
        
        #check the optimizing step found the maximum correctly.
        assert np.isclose(m2.param0,1,atol=0.01), "Failed to find likelihood maximum of test model's param0 while testing CCD"
        assert np.isclose(m2.param1,2,atol=0.01), "Failed to find likelihood maximum of test model's param1 while testing CCD"

        #get the CCD positions and log likelihoods at those locations.
        ccdpos,ccdres = m2.CCD()
        
        #(optimise again as CCD moves the parameter values)
        m2.optimize()

        #The important assert - checking if the CCD positions are in the right places.
        assert np.all(np.isclose(np.sum((ccdpos-np.array([1,2]))**2,1)[1:],1.21,atol=0.01)), "CCD placement error - should be 1.21 from mode"

#####CODE TO COMBINE TODO!

import numpy as np
import matplotlib.pyplot as plt
import GPy
%matplotlib inline
np.set_printoptions(suppress=True, precision=10)
from paramz.transformations import Logexp

class SimpleModel(GPy.Model):
    def __init__(self, name, dims, priors=False):
        super(SimpleModel, self).__init__(name)
        self.params = []
        self.peak_loc = range(1,dims+1,1)
        for i,pos in enumerate(self.peak_loc):
            if priors:
                p = GPy.Param('param%d' % i, 1.0, Logexp())
            else:
                p = GPy.Param('param%d' % i, 1.0)
            self.params.append(p)
            self.link_parameter(p)
    
    def log_likelihood(self):
        like = 0
        for i,pos in enumerate(self.peak_loc):
            like -= ((self.params[i])-pos)**2
        return like[0]
    
    def parameters_changed(self):
        for i,pos in enumerate(self.peak_loc):
            self.params[i].gradient = -2*((self.params[i])-pos)
            
import itertools
def find_likes(m,stepsize=0.3,rangemin=-2,rangemax=7):
    """Numerical grid integral over model parameters
    This function returns an array of all the """
    params = m.parameter_names_flat()
    param_ranges = []
    for param in params:
        param_ranges.append(np.arange(rangemin,rangemax,stepsize))
    combs = itertools.product(*param_ranges)
    llsum = 0
    for el in combs:
        llsum+=np.exp(-m._objective(el))
    return llsum

for dims in range(1,8):
    m1 = SimpleModel('simple',dims)
    m1.optimize()
    assert np.all(np.isclose(m1.numerical_parameter_hessian(),np.eye(dims)*2)), "Numerical approximation to Hessian doesn't match. Error in numerical_parameter_hessian()."
    peak_loc = range(1,dims+1)
    for d in range(dims):        
        assert np.isclose(m1.params[d],d+1,atol=0.01), "Failed to find likelihood maximum of test model's parameter while testing CCD"

    ccdpos,ccdres = m1.CCD()
    m1.optimize()
    dists = np.sum((ccdpos-np.array(peak_loc))**2,1)[1:]
    dists - np.mean(dists)
    assert np.all(np.isclose(dists,np.mean(dists),atol=0.01)), "CCD placement error - should be 1.21 from mode, for nd symmetrical Quadratic test case"
    assert np.all(np.isclose(np.sum(ccdres[1:]/ccdres[0]),4.7619,atol=0.1)), "CCD placement error - off-centre locations should have log likelihood ratios to central point summing to 4.76 times the centre, for nd symmetrical Quadratic test case"
    
for dims in range(1,5):
    stepsize=0.3*dims #make step size bigger as dims goes up so this isn't too slow
    m2 = SimpleModel('simple',dims,True)
    ls = find_likes(m2,stepsize)
    numsum = ls*(stepsize**dims)
    hes = m2.numerical_parameter_hessian()
    hessum = np.exp(m2.log_likelihood())*1/np.sqrt(np.linalg.det(1/(2*np.pi)*hes))
    assert np.isclose(hessum,numsum,atol=0.2), "Laplace approximation using numerical_parameter_hessian()=%0.4f not equal to numerical grid sum=%0.4f" % (hessum,numsum)

#Test numerical_parameter_hessian gives us the right integral for a more complex GP model
#sample data
X = np.arange(0,40,1)[:,None]
Y = np.sin(X/5)+np.random.randn(X.shape[0],X.shape[1])*0.1
k = GPy.kern.RBF(1)

#create model and optimise
m2 = GPy.models.GPRegression(X,Y,k)
m2.Gaussian_noise.fix(0.5)
m2.optimize()

m2.numerical_parameter_hessian()

dims = 2 #equals the number of unfixed parameters
stepsize=0.2
ls = find_likes(m2,stepsize,rangemin=0.0001,rangemax=20)
numsum = ls*(stepsize**dims)
m2.optimize()
hes = m2.numerical_parameter_hessian()
hessum = np.exp(m2.log_likelihood())*1/np.sqrt(np.linalg.det(1/(2*np.pi)*hes))
assert np.isclose(hessum,numsum,atol=0,rtol=0.1), "Laplace approximation using numerical_parameter_hessian() not equal to numerical grid sum"    
