# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from scipy import special
from scipy.special import erfcx
import sys
epsilon = sys.float_info.epsilon
lim_val = -np.log(epsilon) 

def logisticln(x): # pragma: no cover
    return np.where(x<lim_val, np.where(x>-lim_val, -np.log(1+np.exp(-x)), -x), -np.log(1+epsilon))

def logistic(x): # pragma: no cover
    return special.expit(x)
    #return np.where(x<lim_val, np.where(x>-lim_val, 1/(1+np.exp(-x)), epsilon/(epsilon+1)), 1/(1+epsilon))

def normcdf(x): # pragma: no cover
    return special.ndtr(x)
    #g=0.5*erfc(-x/np.sqrt(2))
    #return np.where(g==0, epsilon, np.where(g==1, 1-epsilon, g)) 

def normcdfln(x): # pragma: no cover
    return special.log_ndtr(x)
    #return np.where(x < 0, -.5*x*x + np.log(.5) + np.log(erfcx(-x/np.sqrt(2))), np.log(normcdf(x)))

def clip_exp(x): # pragma: no cover
    return np.where(x<lim_val, np.where(x>-lim_val, np.exp(x), epsilon), 1/epsilon)

def differfln(x0, x1): # pragma: no cover
    # this is a, hopefully!, a numerically more stable variant of log(erf(x0)-erf(x1)) = log(erfc(x1)-erfc(x0)).
    return np.where(x0>x1, -x1*x1 + np.log(erfcx(x1)-np.exp(-x0**2+x1**2)*erfcx(x0)), -x0*x0 + np.log(np.exp(-x1**2+x0**2)*erfcx(x1) - erfcx(x0)))
