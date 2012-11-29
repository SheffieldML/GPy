import numpy as np
from scipy import linalg, optimize
import pylab as pb
import Tango
import sys
import re
import numdifftools as ndt
import pdb
import cPickle


class Metropolis_Hastings:
    def __init__(self,model,cov=None):
        """Metropolis Hastings, with tunings according to Gelman et al. """
        self.model = model
        current = self.model.extract_param()
        self.D = current.size
        self.chains = []
        if cov is None:
            self.cov = model.Laplace_covariance()
        else:
            self.cov = cov
        self.scale = 2.4/np.sqrt(self.D)
        self.new_chain(current)

    def new_chain(self, start=None):
        self.chains.append([])
        if start is None:
            self.model.randomize()
        else:
            self.model.expand_param(start)



    def sample(self, Ntotal, Nburn, Nthin, tune=True, tune_throughout=False, tune_interval=400):
        current = self.model.extract_param()
        fcurrent = self.model.log_likelihood() + self.model.log_prior()
        accepted = np.zeros(Ntotal,dtype=np.bool)
        for it in range(Ntotal):
            print "sample %d of %d\r"%(it,Ntotal),
            sys.stdout.flush()
            prop = np.random.multivariate_normal(current, self.cov*self.scale*self.scale)
            self.model.expand_param(prop)
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
        param = self.model.get_param()
        fs = []
        for p in self.chain:
            self.model.set_param(p)
            fs.append(function(*args))
        self.model.set_param(param)# reset model to starting state
        return fs



