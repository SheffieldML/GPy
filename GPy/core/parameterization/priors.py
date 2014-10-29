# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy.special import gammaln, digamma
from ...util.linalg import pdinv,tdot,backsub_both_sides
from domains import _REAL, _POSITIVE
import warnings
import weakref

class Prior(object):
    domain = None

    def pdf(self, x):
        return np.exp(self.lnpdf(x))

    def plot(self):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ...plotting.matplot_dep import priors_plots
        priors_plots.univariate_plot(self)

    def __repr__(self, *args, **kwargs):
        return self.__str__()

class Gaussian(Prior):
    """
    Implementation of the univariate Gaussian probability function, coupled with random variables.

    :param mu: mean
    :param sigma: standard deviation

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _REAL
    _instances = []
    def __new__(cls, mu, sigma): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().mu == mu and instance().sigma == sigma:
                    return instance()
        o = super(Prior, cls).__new__(cls, mu, sigma)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sigma2 = np.square(self.sigma)
        self.constant = -0.5 * np.log(2 * np.pi * self.sigma2)

    def __str__(self):
        return "N(" + str(np.round(self.mu)) + ', ' + str(np.round(self.sigma2)) + ')'

    def lnpdf(self, x):
        return self.constant - 0.5 * np.square(x - self.mu) / self.sigma2

    def lnpdf_grad(self, x):
        return -(x - self.mu) / self.sigma2

    def rvs(self, n):
        return np.random.randn(n) * self.sigma + self.mu


class Uniform(Prior):
    domain = _REAL
    _instances = []
    def __new__(cls, lower, upper): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().lower == lower and instance().upper == upper:
                    return instance()
        o = super(Prior, cls).__new__(cls, lower, upper)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, lower, upper):
        self.lower = float(lower)
        self.upper = float(upper)

    def __str__(self):
        return "[" + str(np.round(self.lower)) + ', ' + str(np.round(self.upper)) + ']'

    def lnpdf(self, x):
        region = (x>=self.lower) * (x<=self.upper)
        return region

    def lnpdf_grad(self, x):
        return np.zeros(x.shape)

    def rvs(self, n):
        return np.random.uniform(self.lower, self.upper, size=n)

class LogGaussian(Prior):
    """
    Implementation of the univariate *log*-Gaussian probability function, coupled with random variables.

    :param mu: mean
    :param sigma: standard deviation

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    _instances = []
    def __new__(cls, mu, sigma): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().mu == mu and instance().sigma == sigma:
                    return instance()
        o = super(Prior, cls).__new__(cls, mu, sigma)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sigma2 = np.square(self.sigma)
        self.constant = -0.5 * np.log(2 * np.pi * self.sigma2)

    def __str__(self):
        return "lnN(" + str(np.round(self.mu)) + ', ' + str(np.round(self.sigma2)) + ')'

    def lnpdf(self, x):
        return self.constant - 0.5 * np.square(np.log(x) - self.mu) / self.sigma2 - np.log(x)

    def lnpdf_grad(self, x):
        return -((np.log(x) - self.mu) / self.sigma2 + 1.) / x

    def rvs(self, n):
        return np.exp(np.random.randn(n) * self.sigma + self.mu)


class MultivariateGaussian:
    """
    Implementation of the multivariate Gaussian probability function, coupled with random variables.

    :param mu: mean (N-dimensional array)
    :param var: covariance matrix (NxN)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _REAL
    _instances = []
    def __new__(cls, mu, var): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu == mu) and np.all(instance().var == var):
                    return instance()
        o = super(Prior, cls).__new__(cls, mu, var)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, mu, var):
        self.mu = np.array(mu).flatten()
        self.var = np.array(var)
        assert len(self.var.shape) == 2
        assert self.var.shape[0] == self.var.shape[1]
        assert self.var.shape[0] == self.mu.size
        self.input_dim = self.mu.size
        self.inv, self.hld = pdinv(self.var)
        self.constant = -0.5 * self.input_dim * np.log(2 * np.pi) - self.hld

    def summary(self):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.lnpdf(x))

    def lnpdf(self, x):
        d = x - self.mu
        return self.constant - 0.5 * np.sum(d * np.dot(d, self.inv), 1)

    def lnpdf_grad(self, x):
        d = x - self.mu
        return -np.dot(self.inv, d)

    def rvs(self, n):
        return np.random.multivariate_normal(self.mu, self.var, n)

    def plot(self):
        import sys
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import priors_plots
        priors_plots.multivariate_plot(self)

class Gamma(Prior):
    """
    Implementation of the Gamma probability function, coupled with random variables.

    :param a: shape parameter
    :param b: rate parameter (warning: it's the *inverse* of the scale)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    _instances = []
    def __new__(cls, a, b): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().a == a and instance().b == b:
                    return instance()
        o = super(Prior, cls).__new__(cls, a, b)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        self.constant = -gammaln(self.a) + a * np.log(b)

    def __str__(self):
        return "Ga(" + str(np.round(self.a)) + ', ' + str(np.round(self.b)) + ')'

    def summary(self):
        ret = {"E[x]": self.a / self.b, \
            "E[ln x]": digamma(self.a) - np.log(self.b), \
            "var[x]": self.a / self.b / self.b, \
            "Entropy": gammaln(self.a) - (self.a - 1.) * digamma(self.a) - np.log(self.b) + self.a}
        if self.a > 1:
            ret['Mode'] = (self.a - 1.) / self.b
        else:
            ret['mode'] = np.nan
        return ret

    def lnpdf(self, x):
        return self.constant + (self.a - 1) * np.log(x) - self.b * x

    def lnpdf_grad(self, x):
        return (self.a - 1.) / x - self.b

    def rvs(self, n):
        return np.random.gamma(scale=1. / self.b, shape=self.a, size=n)
    @staticmethod
    def from_EV(E, V):
        """
        Creates an instance of a Gamma Prior  by specifying the Expected value(s)
        and Variance(s) of the distribution.

        :param E: expected value
        :param V: variance
        """
        a = np.square(E) / V
        b = E / V
        return Gamma(a, b)

class InverseGamma(Prior):
    """
    Implementation of the inverse-Gamma probability function, coupled with random variables.

    :param a: shape parameter
    :param b: rate parameter (warning: it's the *inverse* of the scale)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    _instances = []
    def __new__(cls, a, b): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().a == a and instance().b == b:
                    return instance()
        o = super(Prior, cls).__new__(cls, a, b)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()
    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        self.constant = -gammaln(self.a) + a * np.log(b)

    def __str__(self):
        return "iGa(" + str(np.round(self.a)) + ', ' + str(np.round(self.b)) + ')'

    def lnpdf(self, x):
        return self.constant - (self.a + 1) * np.log(x) - self.b / x

    def lnpdf_grad(self, x):
        return -(self.a + 1.) / x + self.b / x ** 2

    def rvs(self, n):
        return 1. / np.random.gamma(scale=1. / self.b, shape=self.a, size=n)

class DGPLVM_KFDA(Prior):
    """
    Implementation of the Discriminative Gaussian Process Latent Variable function using
    Kernel Fisher Discriminant Analysis by Seung-Jean Kim for implementing Face paper
    by Chaochao Lu.

    :param lambdaa: constant
    :param sigma2: constant

    .. Note:: Surpassing Human-Level Face paper dgplvm implementation

    """
    domain = _REAL
    # _instances = []
    # def __new__(cls, lambdaa, sigma2):  # Singleton:
    #     if cls._instances:
    #         cls._instances[:] = [instance for instance in cls._instances if instance()]
    #         for instance in cls._instances:
    #             if instance().mu == mu and instance().sigma == sigma:
    #                 return instance()
    #     o = super(Prior, cls).__new__(cls, mu, sigma)
    #     cls._instances.append(weakref.ref(o))
    #     return cls._instances[-1]()

    def __init__(self, lambdaa, sigma2, lbl, kern, x_shape):
        """A description for init"""
        self.datanum = lbl.shape[0]
        self.classnum = lbl.shape[1]
        self.lambdaa = lambdaa
        self.sigma2 = sigma2
        self.lbl = lbl
        self.kern = kern
        lst_ni = self.compute_lst_ni()
        self.a = self.compute_a(lst_ni)
        self.A = self.compute_A(lst_ni)
        self.x_shape = x_shape

    def get_class_label(self, y):
        for idx, v in enumerate(y):
            if v == 1:
                return idx
        return -1

    # This function assigns each data point to its own class
    # and returns the dictionary which contains the class name and parameters.
    def compute_cls(self, x):
        cls = {}
        # Appending each data point to its proper class
        for j in xrange(self.datanum):
            class_label = self.get_class_label(self.lbl[j])
            if class_label not in cls:
                cls[class_label] = []
            cls[class_label].append(x[j])
        if len(cls) > 2:
            for i in range(2, self.classnum):
                del cls[i]
        return cls

    def x_reduced(self, cls):
        x1 = cls[0]
        x2 = cls[1]
        x = np.concatenate((x1, x2), axis=0)
        return x

    def compute_lst_ni(self):
        lst_ni = []
        lst_ni1 = []
        lst_ni2 = []
        f1 = (np.where(self.lbl[:, 0] == 1)[0])
        f2 = (np.where(self.lbl[:, 1] == 1)[0])
        for idx in f1:
            lst_ni1.append(idx)
        for idx in f2:
            lst_ni2.append(idx)
        lst_ni.append(len(lst_ni1))
        lst_ni.append(len(lst_ni2))
        return lst_ni

    def compute_a(self, lst_ni):
        a = np.ones((self.datanum, 1))
        count = 0
        for N_i in lst_ni:
            if N_i == lst_ni[0]:
                a[count:count + N_i] = (float(1) / N_i) * a[count]
                count += N_i
            else:
                if N_i == lst_ni[1]:
                    a[count: count + N_i] = -(float(1) / N_i) * a[count]
                    count += N_i
        return a

    def compute_A(self, lst_ni):
        A = np.zeros((self.datanum, self.datanum))
        idx = 0
        for N_i in lst_ni:
            B = float(1) / np.sqrt(N_i) * (np.eye(N_i) - ((float(1) / N_i) * np.ones((N_i, N_i))))
            A[idx:idx + N_i, idx:idx + N_i] = B
            idx += N_i
        return A

    # Here log function
    def lnpdf(self, x):
        x = x.reshape(self.x_shape)
        K = self.kern.K(x)
        a_trans = np.transpose(self.a)
        paran = self.lambdaa * np.eye(x.shape[0]) + self.A.dot(K).dot(self.A)
        inv_part = pdinv(paran)[0]
        J = a_trans.dot(K).dot(self.a) - a_trans.dot(K).dot(self.A).dot(inv_part).dot(self.A).dot(K).dot(self.a)
        J_star = (1. / self.lambdaa) * J
        return (-1. / self.sigma2) * J_star

    # Here gradient function
    def lnpdf_grad(self, x):
        x = x.reshape(self.x_shape)
        K = self.kern.K(x)
        paran = self.lambdaa * np.eye(x.shape[0]) + self.A.dot(K).dot(self.A)
        inv_part = pdinv(paran)[0]
        b = self.A.dot(inv_part).dot(self.A).dot(K).dot(self.a)
        a_Minus_b = self.a - b
        a_b_trans = np.transpose(a_Minus_b)
        DJ_star_DK = (1. / self.lambdaa) * (a_Minus_b.dot(a_b_trans))
        DJ_star_DX = self.kern.gradients_X(DJ_star_DK, x)
        return (-1. / self.sigma2) * DJ_star_DX

    def rvs(self, n):
        return np.random.rand(n)  # A WRONG implementation

    def __str__(self):
        return 'DGPLVM_prior'


class DGPLVM(Prior):
    """
    Implementation of the Discriminative Gaussian Process Latent Variable model paper, by Raquel.

    :param sigma2: constant

    .. Note:: DGPLVM for Classification paper implementation

    """
    domain = _REAL

    def __init__(self, sigma2, label, x_shape, jit=0.):
        self.sigma2 = sigma2
        self.labels = np.unique(label)
        self.cls_idx = [np.where(label==l)[0] for l in self.labels]
        self.classnum = self.labels.shape[0]
        self.datanum = x_shape[0]
        self.x_shape = x_shape
        self.cls_ratio = np.array([float(len(idx))/self.datanum for idx in self.cls_idx])
        self.jit = jit

        
    def _compute_SbSw(self,X):
        M_0 = X.mean(axis=0)
        Ms = np.vstack([X[idx].mean(axis=0) for idx in self.cls_idx])
        
        tmp = Ms - M_0
        Sb = np.dot(tmp.T,self.cls_ratio[:,None]*tmp)
        Sw = np.sum([tdot((X[idx]-Ms[i]).T) for i,idx in enumerate(self.cls_idx)],axis=0)/self.datanum
        return Sb,Sw
    
#     def _compute(self,X):
#         X = X.reshape(self.x_shape)
#  
#         M_0 = X.mean(axis=0)
#         Ms = np.vstack([X[idx].mean(axis=0) for idx in self.cls_idx])
#         print Ms
#          
#         dMs = Ms - M_0
#         dX = X.copy()
#         for i,idx in enumerate(self.cls_idx):
#             dX[idx] = X[idx]-Ms[i]
#  
#         Sb = np.dot(dMs.T,self.cls_ratio[:,None]*dMs)
# #         Sb = np.identity(self.x_shape[1])
# #        print Sb
#         Sw = np.sum([tdot(dX[idx].T) for idx in self.cls_idx],axis=0)/self.datanum
#          
#         Swinv,Lw,_,_ = pdinv(Sw+self.jit*np.identity(self.x_shape[1]))
#         LwinvSbLwinvT = backsub_both_sides(Lw,Sb,transpose='right')
# #         lnpdf =  -1./(self.sigma2*np.trace(LwinvSbLwinvT))
#         lnpdf =  np.trace(LwinvSbLwinvT)/self.sigma2
#          
#         SwinvSbSwinv = backsub_both_sides(Lw,LwinvSbLwinvT,transpose='left')
#          
#         dX = -np.dot(dX,SwinvSbSwinv)
#         dMs = np.dot(dMs,Swinv)
#         for i,idx in enumerate(self.cls_idx):
#             dX[idx] += dMs[i]
#          
# #         dX *= 2.*lnpdf*lnpdf*self.sigma2/self.datanum
#         dX *= 2./self.sigma2/self.datanum
#         return lnpdf, dX
        
    def _compute(self,X):
        X = X.reshape(self.x_shape)
 
        M_0 = X.mean(axis=0)
        Ms = np.vstack([X[idx].mean(axis=0) for idx in self.cls_idx])
         
        dMs = Ms - M_0
        dX = X.copy()
        for i,idx in enumerate(self.cls_idx):
            dX[idx] = X[idx]-Ms[i]
 
        Sb = np.dot(dMs.T,self.cls_ratio[:,None]*dMs)
        Sw = np.sum([tdot(dX[idx].T) for idx in self.cls_idx],axis=0)/self.datanum
         
        Sbinv,Lb,_,_ = pdinv(Sb+self.jit*np.identity(self.x_shape[1]))
        LbinvSwLbinvT = backsub_both_sides(Lb,Sw,transpose='right')
        lnpdf =  -np.trace(LbinvSwLbinvT)/self.sigma2
         
        SbinvSwSbinv = backsub_both_sides(Lb,LbinvSwLbinvT,transpose='left')
         
        dX = np.dot(dX,Sbinv)
        dMs = -np.dot(dMs,SbinvSwSbinv)
        for i,idx in enumerate(self.cls_idx):
            dX[idx] += dMs[i]
         
        dX *= -2./self.sigma2/self.datanum
         
        return lnpdf, dX
        
    def lnpdf(self, x):
        lnpdf,_ = self._compute(x)
        return lnpdf

    def lnpdf_grad(self, x):
        _, dX = self._compute(x)
        return dX.flat

    def rvs(self, n):
        return np.random.rand(n)  # A WRONG implementation

    def __str__(self):
        return 'DGPLVM_prior'

    
class Dis_prior(Prior):
    """
    Implementation of the Discriminative Gaussian Process Latent Variable model paper, by Raquel.

    :param sigma2: constant

    .. Note:: DGPLVM for Classification paper implementation

    """
    domain = _REAL

    def __init__(self, sigma2, label, x_shape, jit=0.):
        self.sigma2 = sigma2
        self.labels = np.unique(label)
        self.cls_idx = [np.where(label==l)[0] for l in self.labels]
        self.classnum = self.labels.shape[0]
        self.datanum = x_shape[0]
        self.x_shape = x_shape
        self.cls_ratio = np.array([float(len(idx))/self.datanum for idx in self.cls_idx])
        self.jit = jit
#        self.lengthscale = lengthscale
        self.lengthscale = np.ones(x_shape[1])
        
    def _compute(self,X):
        X = X.reshape(self.x_shape)/self.lengthscale
 
        Ms = np.vstack([X[idx].mean(axis=0) for idx in self.cls_idx])
         
        dX = X.copy()
        for i,idx in enumerate(self.cls_idx):
            dX[idx] = X[idx]-Ms[i]
        
        dMs = (Ms[None,:,:]-Ms[:,None,:])
        dMs_c = dMs.sum(axis=0)
        Ms_num = (self.classnum-1)*self.classnum/2
        
        nom = np.square(dX).sum()/self.datanum
        denom = np.square(dMs).sum()/(2*Ms_num)
        
        lnpdf =  -nom/(denom*self.sigma2)
         
        dX *= -1./(denom*self.datanum)
        for i,idx in enumerate(self.cls_idx):
            dX[idx] += nom/(denom*denom*Ms_num*len(idx))*dMs_c[i]
         
        dX *= 2./(self.sigma2*self.lengthscale)
         
        return lnpdf, dX
        
    def lnpdf(self, x):
        lnpdf,_ = self._compute(x)
        return lnpdf

    def lnpdf_grad(self, x):
        _, dX = self._compute(x)
        return dX.flat

    def rvs(self, n):
        return np.random.rand(n)  # A WRONG implementation

    def __str__(self):
        return 'DGPLVM_prior'
