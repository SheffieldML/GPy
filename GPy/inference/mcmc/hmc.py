# ## Copyright (c) 2014 Mu Niu, Zhenwen Dai and GPy Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np


class HMC:
    """
    An implementation of Hybrid Monte Carlo (HMC) for GPy models
    
    Initialize an object for HMC sampling. Note that the status of the model (model parameters) will be changed during sampling.
    
    :param model: the GPy model that will be sampled
    :type model: GPy.core.Model
    :param M: the mass matrix (an identity matrix by default)
    :type M: numpy.ndarray
    :param stepsize: the step size for HMC sampling
    :type stepsize: float
    """
    def __init__(self, model, M=None,stepsize=1e-1):
        self.model = model
        self.stepsize = stepsize
        self.p = np.empty_like(model.optimizer_array.copy())
        if M is None:
            self.M = np.eye(self.p.size)
        else:
            self.M = M
        self.Minv = np.linalg.inv(self.M)

    def sample(self, num_samples=1000, hmc_iters=20):
        """
        Sample the (unfixed) model parameters.
        
        :param num_samples: the number of samples to draw (1000 by default)
        :type num_samples: int
        :param hmc_iters: the number of leap-frog iterations (20 by default)
        :type hmc_iters: int
        :return: the list of parameters samples with the size N x P (N - the number of samples, P - the number of parameters to sample) 
        :rtype: numpy.ndarray
        """
        params = np.empty((num_samples,self.p.size))
        for i in range(num_samples):
            self.p[:] = np.random.multivariate_normal(np.zeros(self.p.size),self.M)
            H_old = self._computeH()
            theta_old = self.model.optimizer_array.copy()
            params[i] = self.model.unfixed_param_array
            #Matropolis
            self._update(hmc_iters)
            H_new = self._computeH()

            if H_old>H_new:
                k = 1.
            else:
                k = np.exp(H_old-H_new)
            if np.random.rand()<k:
                params[i] = self.model.unfixed_param_array
            else:
                self.model.optimizer_array = theta_old
        return params

    def _update(self, hmc_iters):
        for i in range(hmc_iters):
            self.p[:] += -self.stepsize/2.*self.model._transform_gradients(self.model.objective_function_gradients())
            self.model.optimizer_array = self.model.optimizer_array + self.stepsize*np.dot(self.Minv, self.p)
            self.p[:] += -self.stepsize/2.*self.model._transform_gradients(self.model.objective_function_gradients())

    def _computeH(self,):
        return self.model.objective_function()+self.p.size*np.log(2*np.pi)/2.+np.log(np.linalg.det(self.M))/2.+np.dot(self.p, np.dot(self.Minv,self.p[:,None]))/2.

class HMC_shortcut:
    def __init__(self,model,M=None,stepsize_range=[1e-6, 1e-1],groupsize=5, Hstd_th=[1e-5, 3.]):
        self.model = model
        self.stepsize_range = np.log(stepsize_range)
        self.p = np.empty_like(model.optimizer_array.copy())
        self.groupsize = groupsize
        self.Hstd_th = Hstd_th
        if M is None:
            self.M = np.eye(self.p.size)
        else:
            self.M = M
        self.Minv = np.linalg.inv(self.M)

    def sample(self, m_iters=1000, hmc_iters=20):
        params = np.empty((m_iters,self.p.size))
        for i in range(m_iters):
            # sample a stepsize from the uniform distribution
            stepsize = np.exp(np.random.rand()*(self.stepsize_range[1]-self.stepsize_range[0])+self.stepsize_range[0])
            self.p[:] = np.random.multivariate_normal(np.zeros(self.p.size),self.M)
            H_old = self._computeH()
            params[i] = self.model.unfixed_param_array
            theta_old = self.model.optimizer_array.copy()
            #Matropolis
            self._update(hmc_iters, stepsize)
            H_new = self._computeH()

            if H_old>H_new:
                k = 1.
            else:
                k = np.exp(H_old-H_new)
            if np.random.rand()<k:
                params[i] = self.model.unfixed_param_array
            else:
                self.model.optimizer_array = theta_old
        return params

    def _update(self, hmc_iters, stepsize):
        theta_buf = np.empty((2*hmc_iters+1,self.model.optimizer_array.size))
        p_buf = np.empty((2*hmc_iters+1,self.p.size))
        H_buf = np.empty((2*hmc_iters+1,))
        # Set initial position
        theta_buf[hmc_iters] = self.model.optimizer_array
        p_buf[hmc_iters] = self.p
        H_buf[hmc_iters] = self._computeH()

        reversal = []
        pos = 1
        i=0
        while i<hmc_iters:
            self.p[:] += -stepsize/2.*self.model._transform_gradients(self.model.objective_function_gradients())
            self.model.optimizer_array = self.model.optimizer_array + stepsize*np.dot(self.Minv, self.p)
            self.p[:] += -stepsize/2.*self.model._transform_gradients(self.model.objective_function_gradients())

            theta_buf[hmc_iters+pos] = self.model.optimizer_array
            p_buf[hmc_iters+pos] = self.p
            H_buf[hmc_iters+pos] = self._computeH()
            i+=1

            if i<self.groupsize:
                pos += 1
                continue
            else:
                if len(reversal)==0:
                    Hlist = range(hmc_iters+pos,hmc_iters+pos-self.groupsize,-1)
                    if self._testH(H_buf[Hlist]):
                        pos += 1
                    else:
                        # Reverse the trajectory for the 1st time
                        reversal.append(pos)
                        if hmc_iters-i>pos:
                            pos = -1
                            i += pos
                            self.model.optimizer_array = theta_buf[hmc_iters]
                            self.p[:] = -p_buf[hmc_iters]
                        else:
                            pos_new = pos-hmc_iters+i
                            self.model.optimizer_array = theta_buf[hmc_iters+pos_new]
                            self.p[:] = -p_buf[hmc_iters+pos_new]
                            break
                else:
                    Hlist = range(hmc_iters+pos,hmc_iters+pos+self.groupsize)

                    if self._testH(H_buf[Hlist]):
                        pos += -1
                    else:
                        # Reverse the trajectory for the 2nd time
                        r = (hmc_iters - i)%((reversal[0]-pos)*2)
                        if r>(reversal[0]-pos):
                            pos_new = 2*reversal[0] - r - pos
                        else:
                            pos_new = pos + r
                        self.model.optimizer_array = theta_buf[hmc_iters+pos_new]
                        self.p[:] = p_buf[hmc_iters+pos_new] # the sign of momentum might be wrong!
                        break

    def _testH(self, Hlist):
        Hstd = np.std(Hlist)
        if Hstd<self.Hstd_th[0] or Hstd>self.Hstd_th[1]:
            return False
        else:
            return True

    def _computeH(self,):
        return self.model.objective_function()+self.p.size*np.log(2*np.pi)/2.+np.log(np.linalg.det(self.M))/2.+np.dot(self.p, np.dot(self.Minv,self.p[:,None]))/2.

