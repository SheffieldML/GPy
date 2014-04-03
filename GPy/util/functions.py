import numpy as np
from scipy.special import erf, erfcx
import sys
epsilon = sys.float_info.epsilon
lim_val = -np.log(epsilon) 

def cum_gaussian(x):
    g=0.5*(1+erf(x/np.sqrt(2)))
    return np.where(g==0, epsilon, np.where(g==1, 1-epsilon, g)) 

def ln_cum_gaussian(x):
    return np.where(x < 0, -.5*x*x + np.log(.5) + np.log(erfcx(-np.sqrt(2)/2*x)), np.log(cum_gaussian(x)))

def clip_exp(x):
    if any(x>=lim_val) or any(x<=-lim_val):
        return np.where(x<lim_val, np.where(x>-lim_val, np.exp(x), np.exp(-lim_val)), np.exp(lim_val))
    else:
        return np.exp(x)
