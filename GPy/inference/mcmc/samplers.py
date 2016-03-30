# ## Copyright (c) 2014, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)
from __future__ import print_function

import numpy as np
import sys


try:
    #In Python 2, cPickle is faster. It does not exist in Python 3 but the underlying code is always used
    #if available
    import cPickle as pickle
except ImportError:
    import pickle


class Metropolis_Hastings(object):
    def __init__(self,model,cov=None):
        """Metropolis Hastings, with tunings according to Gelman et al. """
        self.model = model
        current = self.model.optimizer_array
        self.D = current.size
        self.chains = []
        if cov is None:
            self.cov = np.eye(self.D)
        else:
            self.cov = cov
        self.scale = 2.4/np.sqrt(self.D)
        self.new_chain(current)

    def new_chain(self, start=None):
        self.chains.append([])
        if start is None:
            self.model.randomize()
        else:
            self.model.optimizer_array = start

    def sample(self, Ntotal=10000, Nburn=1000, Nthin=10, tune=True, tune_throughout=False, tune_interval=400):
        current = self.model.optimizer_array
        fcurrent = self.model.log_likelihood() + self.model.log_prior() 
        accepted = np.zeros(Ntotal,dtype=np.bool)
        for it in range(Ntotal):
            print("sample %d of %d\r"%(it+1,Ntotal),end="")
            sys.stdout.flush()
            prop = np.random.multivariate_normal(current, self.cov*self.scale*self.scale)
            self.model.optimizer_array = prop
            fprop = self.model.log_likelihood() + self.model.log_prior() 

            if fprop>fcurrent:#sample accepted, going 'uphill'
                accepted[it] = True
                current = prop
                fcurrent = fprop
            else:
                u = np.random.rand()
                if np.exp(fprop-fcurrent)>u:#sample accepted downhill
                    accepted[it] = True
                    current = prop
                    fcurrent = fprop

            #store current value
            if (it > Nburn) & ((it%Nthin)==0):
                self.chains[-1].append(current)

            #tuning!
            if it & ((it%tune_interval)==0) & tune & ((it<Nburn) | (tune_throughout)):
                pc = np.mean(accepted[it-tune_interval:it])
                self.cov = np.cov(np.vstack(self.chains[-1][-tune_interval:]).T)
                if pc > .25:
                    self.scale *= 1.1
                if pc < .15:
                    self.scale /= 1.1

    def predict(self,function,args):
        """Make a prediction for the function, to which we will pass the additional arguments"""
        param = self.model.param_array
        fs = []
        for p in self.chain:
            self.model.param_array = p
            fs.append(function(*args))
        # reset model to starting state
        self.model.param_array = param
        return fs
