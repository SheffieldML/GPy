import numpy as np
from scipy.special import erf, erfc, erfcx
import sys
epsilon = sys.float_info.epsilon
lim_val = -np.log(epsilon) 

def logisticln(x):
    return np.where(x<lim_val, np.where(x>-lim_val, -np.log(1+np.exp(-x)), -x), -np.log(1+epsilon))

def logistic(x):
    return np.where(x<lim_val, np.where(x>-lim_val, 1/(1+np.exp(-x)), epsilon/(epsilon+1)), 1/(1+epsilon))

def normcdf(x):
    g=0.5*erfc(-x/np.sqrt(2))
    return np.where(g==0, epsilon, np.where(g==1, 1-epsilon, g)) 

def normcdfln(x):
    return np.where(x < 0, -.5*x*x + np.log(.5) + np.log(erfcx(-x/np.sqrt(2))), np.log(normcdf(x)))

def clip_exp(x):
    return np.where(x<lim_val, np.where(x>-lim_val, np.exp(x), epsilon), 1/epsilon)
