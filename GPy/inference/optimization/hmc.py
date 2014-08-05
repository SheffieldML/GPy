"""HMC implementation"""

import numpy as np


class HMC:
    def __init__(self,model,M=None,stepsize=1e-1):
        self.model = model
        self.stepsize = stepsize
        self.p = np.empty_like(model.optimizer_array.copy())
        if M is None:
            self.M = np.eye(self.p.size)
        else:
            self.M = M
        self.Minv = np.linalg.inv(self.M)

    def sample(self, m_iters=1000, hmc_iters=20):
        thetas = np.empty((m_iters,self.p.size))
        ps = np.empty((m_iters,self.p.size))
        for i in xrange(m_iters):
            #Gibbs
            self.p[:] = np.random.multivariate_normal(np.ones(self.p.size),self.M)
            H_old = self._computeH()
            p_old = self.p.copy()
            theta_old = self.model.optimizer_array.copy()
            #Matropolis
            self._update(hmc_iters)
            H_new = self._computeH()

            k = np.exp(H_old-H_new)
            print k
            if np.random.rand()<k:
                thetas[i] = self.model.optimizer_array
                ps[i] = self.p
            else:
                thetas[i] = theta_old
                ps[i] = p_old
                self.model.optimizer_array = theta_old
        return thetas, ps

    def _update(self, hmc_iters):
        for i in xrange(hmc_iters):
            g = self.p.copy()
            g[:] = 1e-2
#            self.p[:] += self.stepsize/2.*self.model.grad()[:,0]#*-self.model._transform_gradients(self.model.objective_function_gradients())
            self.p[:] += self.stepsize/2.*-self.model._transform_gradients(self.model.objective_function_gradients())
            self.model.optimizer_array[:] = self.model.optimizer_array[:] + self.stepsize*np.dot(self.Minv, self.p[:,None])[:,0]
            self.p[:] += self.stepsize/2.*-self.model._transform_gradients(self.model.objective_function_gradients())
            #self.model.optimizer_array = self.model.optimizer_array - self.stepsize*self.model._transform_gradients(self.model.objective_function_gradients())

    def _computeH(self,):
        return self.model.objective_function()+self.p.size*np.log(2*np.pi)/2.+np.log(np.linalg.det(self.M))/2.+np.dot(self.p, np.dot(self.Minv,self.p[:,None]))/2.

class Gmodel:
    def __init__(self,):
        self.cov = np.array([[1., 0.99],[0.99, 1.]])
        self.optimizer_array = np.random.rand(2)

    def grad(self,):
        return -np.dot(np.linalg.inv(self.cov),self.optimizer_array[:,None])

    def objective_function(self,):
        return np.dot(self.optimizer_array, np.dot(np.linalg.inv(self.cov),self.optimizer_array[:,None]))/2.
