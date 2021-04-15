# Copyright (c) 2012 - 2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy.special import gammaln, digamma
from ...util.linalg import pdinv
from paramz.domains import _REAL, _POSITIVE, _NEGATIVE
import warnings
import weakref


class Prior(object):
    domain = None
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance or cls._instance.__class__ is not cls:
                newfunc = super(Prior, cls).__new__
                if newfunc is object.__new__:
                    cls._instance = newfunc(cls)
                else:
                    cls._instance = newfunc(cls, *args, **kwargs)
                return cls._instance

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

    def __new__(cls, mu=0, sigma=1):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().mu == mu and instance().sigma == sigma:
                    return instance()
        newfunc = super(Prior, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)
        else:
            o = newfunc(cls, mu, sigma)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sigma2 = np.square(self.sigma)
        self.constant = -0.5 * np.log(2 * np.pi * self.sigma2)

    def __str__(self):
        return "N({:.2g}, {:.2g})".format(self.mu, self.sigma)

    def lnpdf(self, x):
        return self.constant - 0.5 * np.square(x - self.mu) / self.sigma2

    def lnpdf_grad(self, x):
        return -(x - self.mu) / self.sigma2

    def rvs(self, n):
        return np.random.randn(n) * self.sigma + self.mu

#     def __getstate__(self):
#         return self.mu, self.sigma
#
#     def __setstate__(self, state):
#         self.mu = state[0]
#         self.sigma = state[1]
#         self.sigma2 = np.square(self.sigma)
#         self.constant = -0.5 * np.log(2 * np.pi * self.sigma2)

class Uniform(Prior):
    _instances = []

    def __new__(cls, lower=0, upper=1):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().lower == lower and instance().upper == upper:
                    return instance()
        newfunc = super(Prior, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)
        else:
            o = newfunc(cls, lower, upper)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, lower, upper):
        self.lower = float(lower)
        self.upper = float(upper)
        assert self.lower < self.upper, "Lower needs to be strictly smaller than upper."
        if self.lower >= 0:
            self.domain = _POSITIVE
        elif self.upper <= 0:
            self.domain = _NEGATIVE
        else:
            self.domain = _REAL

    def __str__(self):
        return "[{:.2g}, {:.2g}]".format(self.lower, self.upper)

    def lnpdf(self, x):
        region = (x >= self.lower) * (x <= self.upper)
        return region

    def lnpdf_grad(self, x):
        return np.zeros(x.shape)

    def rvs(self, n):
        return np.random.uniform(self.lower, self.upper, size=n)

#     def __getstate__(self):
#         return self.lower, self.upper
#
#     def __setstate__(self, state):
#         self.lower = state[0]
#         self.upper = state[1]

class LogGaussian(Gaussian):
    """
    Implementation of the univariate *log*-Gaussian probability function, coupled with random variables.

    :param mu: mean
    :param sigma: standard deviation

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    _instances = []

    def __new__(cls, mu=0, sigma=1):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().mu == mu and instance().sigma == sigma:
                    return instance()
        newfunc = super(Prior, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)  
        else:
            o = newfunc(cls, mu, sigma)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sigma2 = np.square(self.sigma)
        self.constant = -0.5 * np.log(2 * np.pi * self.sigma2)

    def __str__(self):
        return "lnN({:.2g}, {:.2g})".format(self.mu, self.sigma)

    def lnpdf(self, x):
        return self.constant - 0.5 * np.square(np.log(x) - self.mu) / self.sigma2 - np.log(x)

    def lnpdf_grad(self, x):
        return -((np.log(x) - self.mu) / self.sigma2 + 1.) / x

    def rvs(self, n):
        return np.exp(np.random.randn(int(n)) * self.sigma + self.mu)


class MultivariateGaussian(Prior):
    """
    Implementation of the multivariate Gaussian probability function, coupled with random variables.

    :param mu: mean (N-dimensional array)
    :param var: covariance matrix (NxN)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _REAL
    _instances = []

    def __new__(cls, mu=0, var=1):  # Singleton:
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

    def __getstate__(self):
        return self.mu, self.var

    def __setstate__(self, state):
        self.mu = state[0]
        self.var = state[1]
        assert len(self.var.shape) == 2
        assert self.var.shape[0] == self.var.shape[1]
        assert self.var.shape[0] == self.mu.size
        self.input_dim = self.mu.size
        self.inv, self.hld = pdinv(self.var)
        self.constant = -0.5 * self.input_dim * np.log(2 * np.pi) - self.hld

def gamma_from_EV(E, V):
    warnings.warn("use Gamma.from_EV to create Gamma Prior", FutureWarning)
    return Gamma.from_EV(E, V)


class Gamma(Prior):
    """
    Implementation of the Gamma probability function, coupled with random variables.

    :param a: shape parameter
    :param b: rate parameter (warning: it's the *inverse* of the scale)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    _instances = []

    def __new__(cls, a=1, b=.5):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().a == a and instance().b == b:
                    return instance()
        newfunc = super(Prior, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)
        else:
            o = newfunc(cls, a, b)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def __init__(self, a, b):
        self._a = float(a)
        self._b = float(b)
        self.constant = -gammaln(self.a) + a * np.log(b)

    def __str__(self):
        return "Ga({:.2g}, {:.2g})".format(self.a, self.b)

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

    def __getstate__(self):
        return self.a, self.b

    def __setstate__(self, state):
        self._a = state[0]
        self._b = state[1]
        self.constant = -gammaln(self.a) + self.a * np.log(self.b)

class InverseGamma(Gamma):
    """
    Implementation of the inverse-Gamma probability function, coupled with random variables.

    :param a: shape parameter
    :param b: rate parameter (warning: it's the *inverse* of the scale)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    _instances = []

    def __init__(self, a, b):
        self._a = float(a)
        self._b = float(b)
        self.constant = -gammaln(self.a) + a * np.log(b)

    def __str__(self):
        return "iGa({:.2g}, {:.2g})".format(self.a, self.b)

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
        for j in range(self.datanum):
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

    def __getstate___(self):
        return self.lbl, self.lambdaa, self.sigma2, self.kern, self.x_shape

    def __setstate__(self, state):
        lbl, lambdaa, sigma2, kern, a, A, x_shape = state
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


class DGPLVM(Prior):
    """
    Implementation of the Discriminative Gaussian Process Latent Variable model paper, by Raquel.

    :param sigma2: constant

    .. Note:: DGPLVM for Classification paper implementation

    """
    domain = _REAL

    def __new__(cls, sigma2, lbl, x_shape):
        return super(Prior, cls).__new__(cls, sigma2, lbl, x_shape)

    def __init__(self, sigma2, lbl, x_shape):
        self.sigma2 = sigma2
        # self.x = x
        self.lbl = lbl
        self.classnum = lbl.shape[1]
        self.datanum = lbl.shape[0]
        self.x_shape = x_shape
        self.dim = x_shape[1]

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
        for j in range(self.datanum):
            class_label = self.get_class_label(self.lbl[j])
            if class_label not in cls:
                cls[class_label] = []
            cls[class_label].append(x[j])
        return cls

    # This function computes mean of each class. The mean is calculated through each dimension
    def compute_Mi(self, cls):
        M_i = np.zeros((self.classnum, self.dim))
        for i in cls:
            # Mean of each class
            class_i = cls[i]
            M_i[i] = np.mean(class_i, axis=0)
        return M_i

    # Adding data points as tuple to the dictionary so that we can access indices
    def compute_indices(self, x):
        data_idx = {}
        for j in range(self.datanum):
            class_label = self.get_class_label(self.lbl[j])
            if class_label not in data_idx:
                data_idx[class_label] = []
            t = (j, x[j])
            data_idx[class_label].append(t)
        return data_idx

    # Adding indices to the list so we can access whole the indices
    def compute_listIndices(self, data_idx):
        lst_idx = []
        lst_idx_all = []
        for i in data_idx:
            if len(lst_idx) == 0:
                pass
                #Do nothing, because it is the first time list is created so is empty
            else:
                lst_idx = []
            # Here we put indices of each class in to the list called lst_idx_all
            for m in range(len(data_idx[i])):
                lst_idx.append(data_idx[i][m][0])
            lst_idx_all.append(lst_idx)
        return lst_idx_all

    # This function calculates between classes variances
    def compute_Sb(self, cls, M_i, M_0):
        Sb = np.zeros((self.dim, self.dim))
        for i in cls:
            B = (M_i[i] - M_0).reshape(self.dim, 1)
            B_trans = B.transpose()
            Sb += (float(len(cls[i])) / self.datanum) * B.dot(B_trans)
        return Sb

    # This function calculates within classes variances
    def compute_Sw(self, cls, M_i):
        Sw = np.zeros((self.dim, self.dim))
        for i in cls:
            N_i = float(len(cls[i]))
            W_WT = np.zeros((self.dim, self.dim))
            for xk in cls[i]:
                W = (xk - M_i[i])
                W_WT += np.outer(W, W)
            Sw += (N_i / self.datanum) * ((1. / N_i) * W_WT)
        return Sw

    # Calculating beta and Bi for Sb
    def compute_sig_beta_Bi(self, data_idx, M_i, M_0, lst_idx_all):
        # import pdb
        # pdb.set_trace()
        B_i = np.zeros((self.classnum, self.dim))
        Sig_beta_B_i_all = np.zeros((self.datanum, self.dim))
        for i in data_idx:
            # pdb.set_trace()
            # Calculating Bi
            B_i[i] = (M_i[i] - M_0).reshape(1, self.dim)
        for k in range(self.datanum):
            for i in data_idx:
                N_i = float(len(data_idx[i]))
                if k in lst_idx_all[i]:
                    beta = (float(1) / N_i) - (float(1) / self.datanum)
                    Sig_beta_B_i_all[k] += float(N_i) / self.datanum * (beta * B_i[i])
                else:
                    beta = -(float(1) / self.datanum)
                    Sig_beta_B_i_all[k] += float(N_i) / self.datanum * (beta * B_i[i])
        Sig_beta_B_i_all = Sig_beta_B_i_all.transpose()
        return Sig_beta_B_i_all


    # Calculating W_j s separately so we can access all the W_j s anytime
    def compute_wj(self, data_idx, M_i):
        W_i = np.zeros((self.datanum, self.dim))
        for i in data_idx:
            N_i = float(len(data_idx[i]))
            for tpl in data_idx[i]:
                xj = tpl[1]
                j = tpl[0]
                W_i[j] = (xj - M_i[i])
        return W_i

    # Calculating alpha and Wj for Sw
    def compute_sig_alpha_W(self, data_idx, lst_idx_all, W_i):
        Sig_alpha_W_i = np.zeros((self.datanum, self.dim))
        for i in data_idx:
            N_i = float(len(data_idx[i]))
            for tpl in data_idx[i]:
                k = tpl[0]
                for j in lst_idx_all[i]:
                    if k == j:
                        alpha = 1 - (float(1) / N_i)
                        Sig_alpha_W_i[k] += (alpha * W_i[j])
                    else:
                        alpha = 0 - (float(1) / N_i)
                        Sig_alpha_W_i[k] += (alpha * W_i[j])
        Sig_alpha_W_i = (1. / self.datanum) * np.transpose(Sig_alpha_W_i)
        return Sig_alpha_W_i

    # This function calculates log of our prior
    def lnpdf(self, x):
        x = x.reshape(self.x_shape)
        cls = self.compute_cls(x)
        M_0 = np.mean(x, axis=0)
        M_i = self.compute_Mi(cls)
        Sb = self.compute_Sb(cls, M_i, M_0)
        Sw = self.compute_Sw(cls, M_i)
        # sb_N = np.linalg.inv(Sb + np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))
        #Sb_inv_N = np.linalg.inv(Sb+np.eye(Sb.shape[0])*0.1)
        #Sb_inv_N = pdinv(Sb+ np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))[0]
        Sb_inv_N = pdinv(Sb + np.eye(Sb.shape[0])*0.1)[0]
        return (-1 / self.sigma2) * np.trace(Sb_inv_N.dot(Sw))

    # This function calculates derivative of the log of prior function
    def lnpdf_grad(self, x):
        x = x.reshape(self.x_shape)
        cls = self.compute_cls(x)
        M_0 = np.mean(x, axis=0)
        M_i = self.compute_Mi(cls)
        Sb = self.compute_Sb(cls, M_i, M_0)
        Sw = self.compute_Sw(cls, M_i)
        data_idx = self.compute_indices(x)
        lst_idx_all = self.compute_listIndices(data_idx)
        Sig_beta_B_i_all = self.compute_sig_beta_Bi(data_idx, M_i, M_0, lst_idx_all)
        W_i = self.compute_wj(data_idx, M_i)
        Sig_alpha_W_i = self.compute_sig_alpha_W(data_idx, lst_idx_all, W_i)

        # Calculating inverse of Sb and its transpose and minus
        # Sb_inv_N = np.linalg.inv(Sb + np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))
        #Sb_inv_N = np.linalg.inv(Sb+np.eye(Sb.shape[0])*0.1)
        #Sb_inv_N = pdinv(Sb+ np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))[0]
        Sb_inv_N = pdinv(Sb + np.eye(Sb.shape[0])*0.1)[0]
        Sb_inv_N_trans = np.transpose(Sb_inv_N)
        Sb_inv_N_trans_minus = -1 * Sb_inv_N_trans
        Sw_trans = np.transpose(Sw)

        # Calculating DJ/DXk
        DJ_Dxk = 2 * (
            Sb_inv_N_trans_minus.dot(Sw_trans).dot(Sb_inv_N_trans).dot(Sig_beta_B_i_all) + Sb_inv_N_trans.dot(
                Sig_alpha_W_i))
        # Calculating derivative of the log of the prior
        DPx_Dx = ((-1 / self.sigma2) * DJ_Dxk)
        return DPx_Dx.T

    # def frb(self, x):
    #     from functools import partial
    #     from GPy.models import GradientChecker
    #     f = partial(self.lnpdf)
    #     df = partial(self.lnpdf_grad)
    #     grad = GradientChecker(f, df, x, 'X')
    #     grad.checkgrad(verbose=1)

    def rvs(self, n):
        return np.random.rand(n)  # A WRONG implementation

    def __str__(self):
        return 'DGPLVM_prior_Raq'


# ******************************************

from . import Parameterized
from . import Param

class DGPLVM_Lamda(Prior, Parameterized):
    """
    Implementation of the Discriminative Gaussian Process Latent Variable model paper, by Raquel.

    :param sigma2: constant

    .. Note:: DGPLVM for Classification paper implementation

    """
    domain = _REAL
    # _instances = []
    # def __new__(cls, mu, sigma): # Singleton:
    #     if cls._instances:
    #         cls._instances[:] = [instance for instance in cls._instances if instance()]
    #         for instance in cls._instances:
    #             if instance().mu == mu and instance().sigma == sigma:
    #                 return instance()
    #     o = super(Prior, cls).__new__(cls, mu, sigma)
    #     cls._instances.append(weakref.ref(o))
    #     return cls._instances[-1]()

    def __init__(self, sigma2, lbl, x_shape, lamda, name='DP_prior'):
        super(DGPLVM_Lamda, self).__init__(name=name)
        self.sigma2 = sigma2
        # self.x = x
        self.lbl = lbl
        self.lamda = lamda
        self.classnum = lbl.shape[1]
        self.datanum = lbl.shape[0]
        self.x_shape = x_shape
        self.dim = x_shape[1]
        self.lamda = Param('lamda', np.diag(lamda))
        self.link_parameter(self.lamda)

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
        for j in range(self.datanum):
            class_label = self.get_class_label(self.lbl[j])
            if class_label not in cls:
                cls[class_label] = []
            cls[class_label].append(x[j])
        return cls

    # This function computes mean of each class. The mean is calculated through each dimension
    def compute_Mi(self, cls):
        M_i = np.zeros((self.classnum, self.dim))
        for i in cls:
            # Mean of each class
            class_i = cls[i]
            M_i[i] = np.mean(class_i, axis=0)
        return M_i

    # Adding data points as tuple to the dictionary so that we can access indices
    def compute_indices(self, x):
        data_idx = {}
        for j in range(self.datanum):
            class_label = self.get_class_label(self.lbl[j])
            if class_label not in data_idx:
                data_idx[class_label] = []
            t = (j, x[j])
            data_idx[class_label].append(t)
        return data_idx

    # Adding indices to the list so we can access whole the indices
    def compute_listIndices(self, data_idx):
        lst_idx = []
        lst_idx_all = []
        for i in data_idx:
            if len(lst_idx) == 0:
                pass
                #Do nothing, because it is the first time list is created so is empty
            else:
                lst_idx = []
            # Here we put indices of each class in to the list called lst_idx_all
            for m in range(len(data_idx[i])):
                lst_idx.append(data_idx[i][m][0])
            lst_idx_all.append(lst_idx)
        return lst_idx_all

    # This function calculates between classes variances
    def compute_Sb(self, cls, M_i, M_0):
        Sb = np.zeros((self.dim, self.dim))
        for i in cls:
            B = (M_i[i] - M_0).reshape(self.dim, 1)
            B_trans = B.transpose()
            Sb += (float(len(cls[i])) / self.datanum) * B.dot(B_trans)
        return Sb

    # This function calculates within classes variances
    def compute_Sw(self, cls, M_i):
        Sw = np.zeros((self.dim, self.dim))
        for i in cls:
            N_i = float(len(cls[i]))
            W_WT = np.zeros((self.dim, self.dim))
            for xk in cls[i]:
                W = (xk - M_i[i])
                W_WT += np.outer(W, W)
            Sw += (N_i / self.datanum) * ((1. / N_i) * W_WT)
        return Sw

    # Calculating beta and Bi for Sb
    def compute_sig_beta_Bi(self, data_idx, M_i, M_0, lst_idx_all):
        # import pdb
        # pdb.set_trace()
        B_i = np.zeros((self.classnum, self.dim))
        Sig_beta_B_i_all = np.zeros((self.datanum, self.dim))
        for i in data_idx:
            # pdb.set_trace()
            # Calculating Bi
            B_i[i] = (M_i[i] - M_0).reshape(1, self.dim)
        for k in range(self.datanum):
            for i in data_idx:
                N_i = float(len(data_idx[i]))
                if k in lst_idx_all[i]:
                    beta = (float(1) / N_i) - (float(1) / self.datanum)
                    Sig_beta_B_i_all[k] += float(N_i) / self.datanum * (beta * B_i[i])
                else:
                    beta = -(float(1) / self.datanum)
                    Sig_beta_B_i_all[k] += float(N_i) / self.datanum * (beta * B_i[i])
        Sig_beta_B_i_all = Sig_beta_B_i_all.transpose()
        return Sig_beta_B_i_all


    # Calculating W_j s separately so we can access all the W_j s anytime
    def compute_wj(self, data_idx, M_i):
        W_i = np.zeros((self.datanum, self.dim))
        for i in data_idx:
            N_i = float(len(data_idx[i]))
            for tpl in data_idx[i]:
                xj = tpl[1]
                j = tpl[0]
                W_i[j] = (xj - M_i[i])
        return W_i

    # Calculating alpha and Wj for Sw
    def compute_sig_alpha_W(self, data_idx, lst_idx_all, W_i):
        Sig_alpha_W_i = np.zeros((self.datanum, self.dim))
        for i in data_idx:
            N_i = float(len(data_idx[i]))
            for tpl in data_idx[i]:
                k = tpl[0]
                for j in lst_idx_all[i]:
                    if k == j:
                        alpha = 1 - (float(1) / N_i)
                        Sig_alpha_W_i[k] += (alpha * W_i[j])
                    else:
                        alpha = 0 - (float(1) / N_i)
                        Sig_alpha_W_i[k] += (alpha * W_i[j])
        Sig_alpha_W_i = (1. / self.datanum) * np.transpose(Sig_alpha_W_i)
        return Sig_alpha_W_i

    # This function calculates log of our prior
    def lnpdf(self, x):
        x = x.reshape(self.x_shape)

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #self.lamda.values[:] = self.lamda.values/self.lamda.values.sum()

        xprime = x.dot(np.diagflat(self.lamda))
        x = xprime
        # print x
        cls = self.compute_cls(x)
        M_0 = np.mean(x, axis=0)
        M_i = self.compute_Mi(cls)
        Sb = self.compute_Sb(cls, M_i, M_0)
        Sw = self.compute_Sw(cls, M_i)
        # Sb_inv_N = np.linalg.inv(Sb + np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))
        #Sb_inv_N = np.linalg.inv(Sb+np.eye(Sb.shape[0])*0.1)
        #Sb_inv_N = pdinv(Sb+ np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.5))[0]
        Sb_inv_N = pdinv(Sb + np.eye(Sb.shape[0])*0.9)[0]
        return (-1 / self.sigma2) * np.trace(Sb_inv_N.dot(Sw))

    # This function calculates derivative of the log of prior function
    def lnpdf_grad(self, x):
        x = x.reshape(self.x_shape)
        xprime = x.dot(np.diagflat(self.lamda))
        x = xprime
        # print x
        cls = self.compute_cls(x)
        M_0 = np.mean(x, axis=0)
        M_i = self.compute_Mi(cls)
        Sb = self.compute_Sb(cls, M_i, M_0)
        Sw = self.compute_Sw(cls, M_i)
        data_idx = self.compute_indices(x)
        lst_idx_all = self.compute_listIndices(data_idx)
        Sig_beta_B_i_all = self.compute_sig_beta_Bi(data_idx, M_i, M_0, lst_idx_all)
        W_i = self.compute_wj(data_idx, M_i)
        Sig_alpha_W_i = self.compute_sig_alpha_W(data_idx, lst_idx_all, W_i)

        # Calculating inverse of Sb and its transpose and minus
        # Sb_inv_N = np.linalg.inv(Sb + np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))
        #Sb_inv_N = np.linalg.inv(Sb+np.eye(Sb.shape[0])*0.1)
        #Sb_inv_N = pdinv(Sb+ np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.5))[0]
        Sb_inv_N = pdinv(Sb + np.eye(Sb.shape[0])*0.9)[0]
        Sb_inv_N_trans = np.transpose(Sb_inv_N)
        Sb_inv_N_trans_minus = -1 * Sb_inv_N_trans
        Sw_trans = np.transpose(Sw)

        # Calculating DJ/DXk
        DJ_Dxk = 2 * (
            Sb_inv_N_trans_minus.dot(Sw_trans).dot(Sb_inv_N_trans).dot(Sig_beta_B_i_all) + Sb_inv_N_trans.dot(
                Sig_alpha_W_i))
        # Calculating derivative of the log of the prior
        DPx_Dx = ((-1 / self.sigma2) * DJ_Dxk)

        DPxprim_Dx = np.diagflat(self.lamda).dot(DPx_Dx)

        # Because of the GPy we need to transpose our matrix so that it gets the same shape as out matrix (denominator layout!!!)
        DPxprim_Dx = DPxprim_Dx.T

        DPxprim_Dlamda = DPx_Dx.dot(x)

        # Because of the GPy we need to transpose our matrix so that it gets the same shape as out matrix (denominator layout!!!)
        DPxprim_Dlamda = DPxprim_Dlamda.T

        self.lamda.gradient = np.diag(DPxprim_Dlamda)
        # print DPxprim_Dx
        return DPxprim_Dx


    # def frb(self, x):
    #     from functools import partial
    #     from GPy.models import GradientChecker
    #     f = partial(self.lnpdf)
    #     df = partial(self.lnpdf_grad)
    #     grad = GradientChecker(f, df, x, 'X')
    #     grad.checkgrad(verbose=1)

    def rvs(self, n):
        return np.random.rand(n)  # A WRONG implementation

    def __str__(self):
        return 'DGPLVM_prior_Raq_Lamda'

# ******************************************

class DGPLVM_T(Prior):
    """
    Implementation of the Discriminative Gaussian Process Latent Variable model paper, by Raquel.

    :param sigma2: constant

    .. Note:: DGPLVM for Classification paper implementation

    """
    domain = _REAL
    # _instances = []
    # def __new__(cls, mu, sigma): # Singleton:
    #     if cls._instances:
    #         cls._instances[:] = [instance for instance in cls._instances if instance()]
    #         for instance in cls._instances:
    #             if instance().mu == mu and instance().sigma == sigma:
    #                 return instance()
    #     o = super(Prior, cls).__new__(cls, mu, sigma)
    #     cls._instances.append(weakref.ref(o))
    #     return cls._instances[-1]()

    def __init__(self, sigma2, lbl, x_shape, vec):
        self.sigma2 = sigma2
        # self.x = x
        self.lbl = lbl
        self.classnum = lbl.shape[1]
        self.datanum = lbl.shape[0]
        self.x_shape = x_shape
        self.dim = x_shape[1]
        self.vec = vec


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
        for j in range(self.datanum):
            class_label = self.get_class_label(self.lbl[j])
            if class_label not in cls:
                cls[class_label] = []
            cls[class_label].append(x[j])
        return cls

    # This function computes mean of each class. The mean is calculated through each dimension
    def compute_Mi(self, cls):
        M_i = np.zeros((self.classnum, self.dim))
        for i in cls:
            # Mean of each class
            # class_i = np.multiply(cls[i],vec)
            class_i = cls[i]
            M_i[i] = np.mean(class_i, axis=0)
        return M_i

    # Adding data points as tuple to the dictionary so that we can access indices
    def compute_indices(self, x):
        data_idx = {}
        for j in range(self.datanum):
            class_label = self.get_class_label(self.lbl[j])
            if class_label not in data_idx:
                data_idx[class_label] = []
            t = (j, x[j])
            data_idx[class_label].append(t)
        return data_idx

    # Adding indices to the list so we can access whole the indices
    def compute_listIndices(self, data_idx):
        lst_idx = []
        lst_idx_all = []
        for i in data_idx:
            if len(lst_idx) == 0:
                pass
                #Do nothing, because it is the first time list is created so is empty
            else:
                lst_idx = []
            # Here we put indices of each class in to the list called lst_idx_all
            for m in range(len(data_idx[i])):
                lst_idx.append(data_idx[i][m][0])
            lst_idx_all.append(lst_idx)
        return lst_idx_all

    # This function calculates between classes variances
    def compute_Sb(self, cls, M_i, M_0):
        Sb = np.zeros((self.dim, self.dim))
        for i in cls:
            B = (M_i[i] - M_0).reshape(self.dim, 1)
            B_trans = B.transpose()
            Sb += (float(len(cls[i])) / self.datanum) * B.dot(B_trans)
        return Sb

    # This function calculates within classes variances
    def compute_Sw(self, cls, M_i):
        Sw = np.zeros((self.dim, self.dim))
        for i in cls:
            N_i = float(len(cls[i]))
            W_WT = np.zeros((self.dim, self.dim))
            for xk in cls[i]:
                W = (xk - M_i[i])
                W_WT += np.outer(W, W)
            Sw += (N_i / self.datanum) * ((1. / N_i) * W_WT)
        return Sw

    # Calculating beta and Bi for Sb
    def compute_sig_beta_Bi(self, data_idx, M_i, M_0, lst_idx_all):
        # import pdb
        # pdb.set_trace()
        B_i = np.zeros((self.classnum, self.dim))
        Sig_beta_B_i_all = np.zeros((self.datanum, self.dim))
        for i in data_idx:
            # pdb.set_trace()
            # Calculating Bi
            B_i[i] = (M_i[i] - M_0).reshape(1, self.dim)
        for k in range(self.datanum):
            for i in data_idx:
                N_i = float(len(data_idx[i]))
                if k in lst_idx_all[i]:
                    beta = (float(1) / N_i) - (float(1) / self.datanum)
                    Sig_beta_B_i_all[k] += float(N_i) / self.datanum * (beta * B_i[i])
                else:
                    beta = -(float(1) / self.datanum)
                    Sig_beta_B_i_all[k] += float(N_i) / self.datanum * (beta * B_i[i])
        Sig_beta_B_i_all = Sig_beta_B_i_all.transpose()
        return Sig_beta_B_i_all


    # Calculating W_j s separately so we can access all the W_j s anytime
    def compute_wj(self, data_idx, M_i):
        W_i = np.zeros((self.datanum, self.dim))
        for i in data_idx:
            N_i = float(len(data_idx[i]))
            for tpl in data_idx[i]:
                xj = tpl[1]
                j = tpl[0]
                W_i[j] = (xj - M_i[i])
        return W_i

    # Calculating alpha and Wj for Sw
    def compute_sig_alpha_W(self, data_idx, lst_idx_all, W_i):
        Sig_alpha_W_i = np.zeros((self.datanum, self.dim))
        for i in data_idx:
            N_i = float(len(data_idx[i]))
            for tpl in data_idx[i]:
                k = tpl[0]
                for j in lst_idx_all[i]:
                    if k == j:
                        alpha = 1 - (float(1) / N_i)
                        Sig_alpha_W_i[k] += (alpha * W_i[j])
                    else:
                        alpha = 0 - (float(1) / N_i)
                        Sig_alpha_W_i[k] += (alpha * W_i[j])
        Sig_alpha_W_i = (1. / self.datanum) * np.transpose(Sig_alpha_W_i)
        return Sig_alpha_W_i

    # This function calculates log of our prior
    def lnpdf(self, x):
        x = x.reshape(self.x_shape)
        xprim = x.dot(self.vec)
        x = xprim
        # print x
        cls = self.compute_cls(x)
        M_0 = np.mean(x, axis=0)
        M_i = self.compute_Mi(cls)
        Sb = self.compute_Sb(cls, M_i, M_0)
        Sw = self.compute_Sw(cls, M_i)
        # Sb_inv_N = np.linalg.inv(Sb + np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))
        #Sb_inv_N = np.linalg.inv(Sb+np.eye(Sb.shape[0])*0.1)
        #print 'SB_inv: ', Sb_inv_N
        #Sb_inv_N = pdinv(Sb+ np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))[0]
        Sb_inv_N = pdinv(Sb+np.eye(Sb.shape[0])*0.1)[0]
        return (-1 / self.sigma2) * np.trace(Sb_inv_N.dot(Sw))

    # This function calculates derivative of the log of prior function
    def lnpdf_grad(self, x):
        x = x.reshape(self.x_shape)
        xprim = x.dot(self.vec)
        x = xprim
        # print x
        cls = self.compute_cls(x)
        M_0 = np.mean(x, axis=0)
        M_i = self.compute_Mi(cls)
        Sb = self.compute_Sb(cls, M_i, M_0)
        Sw = self.compute_Sw(cls, M_i)
        data_idx = self.compute_indices(x)
        lst_idx_all = self.compute_listIndices(data_idx)
        Sig_beta_B_i_all = self.compute_sig_beta_Bi(data_idx, M_i, M_0, lst_idx_all)
        W_i = self.compute_wj(data_idx, M_i)
        Sig_alpha_W_i = self.compute_sig_alpha_W(data_idx, lst_idx_all, W_i)

        # Calculating inverse of Sb and its transpose and minus
        # Sb_inv_N = np.linalg.inv(Sb + np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))
        #Sb_inv_N = np.linalg.inv(Sb+np.eye(Sb.shape[0])*0.1)
        #print 'SB_inv: ',Sb_inv_N
        #Sb_inv_N = pdinv(Sb+ np.eye(Sb.shape[0]) * (np.diag(Sb).min() * 0.1))[0]
        Sb_inv_N = pdinv(Sb+np.eye(Sb.shape[0])*0.1)[0]
        Sb_inv_N_trans = np.transpose(Sb_inv_N)
        Sb_inv_N_trans_minus = -1 * Sb_inv_N_trans
        Sw_trans = np.transpose(Sw)

        # Calculating DJ/DXk
        DJ_Dxk = 2 * (
            Sb_inv_N_trans_minus.dot(Sw_trans).dot(Sb_inv_N_trans).dot(Sig_beta_B_i_all) + Sb_inv_N_trans.dot(
                Sig_alpha_W_i))
        # Calculating derivative of the log of the prior
        DPx_Dx = ((-1 / self.sigma2) * DJ_Dxk)
        return DPx_Dx.T

    # def frb(self, x):
    #     from functools import partial
    #     from GPy.models import GradientChecker
    #     f = partial(self.lnpdf)
    #     df = partial(self.lnpdf_grad)
    #     grad = GradientChecker(f, df, x, 'X')
    #     grad.checkgrad(verbose=1)

    def rvs(self, n):
        return np.random.rand(n)  # A WRONG implementation

    def __str__(self):
        return 'DGPLVM_prior_Raq_TTT'




class HalfT(Prior):
    """
    Implementation of the half student t probability function, coupled with random variables.

    :param A: scale parameter
    :param nu: degrees of freedom

    """
    domain = _POSITIVE
    _instances = []

    def __new__(cls, A, nu):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().A == A and instance().nu == nu:
                    return instance()
        o = super(Prior, cls).__new__(cls, A, nu)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, A, nu):
        self.A = float(A)
        self.nu = float(nu)
        self.constant = gammaln(.5*(self.nu+1.)) - gammaln(.5*self.nu) - .5*np.log(np.pi*self.A*self.nu)

    def __str__(self):
        return "hT({:.2g}, {:.2g})".format(self.A, self.nu)

    def lnpdf(self, theta):
        return (theta > 0) * (self.constant - .5*(self.nu + 1) * np.log(1. + (1./self.nu) * (theta/self.A)**2))

        # theta = theta if isinstance(theta,np.ndarray) else np.array([theta])
        # lnpdfs = np.zeros_like(theta)
        # theta = np.array([theta])
        # above_zero = theta.flatten()>1e-6
        # v = self.nu
        # sigma2=self.A
        # stop
        # lnpdfs[above_zero] = (+ gammaln((v + 1) * 0.5)
        #     - gammaln(v * 0.5)
        #     - 0.5*np.log(sigma2 * v * np.pi)
        #     - 0.5*(v + 1)*np.log(1 + (1/np.float(v))*((theta[above_zero][0]**2)/sigma2))
        # )
        # return lnpdfs

    def lnpdf_grad(self, theta):
        theta = theta if isinstance(theta, np.ndarray) else np.array([theta])
        grad = np.zeros_like(theta)
        above_zero = theta > 1e-6
        v = self.nu
        sigma2 = self.A
        grad[above_zero] = -0.5*(v+1)*(2*theta[above_zero])/(v*sigma2 + theta[above_zero][0]**2)
        return grad

    def rvs(self, n):
        # return np.random.randn(n) * self.sigma + self.mu
        from scipy.stats import t
        # [np.abs(x) for x in t.rvs(df=4,loc=0,scale=50, size=10000)])
        ret = t.rvs(self.nu, loc=0, scale=self.A, size=n)
        ret[ret < 0] = 0
        return ret


class Exponential(Prior):
    """
    Implementation of the Exponential probability function,
    coupled with random variables.

    :param l: shape parameter

    """
    domain = _POSITIVE
    _instances = []

    def __new__(cls, l):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().l == l:
                    return instance()
        o = super(Exponential, cls).__new__(cls, l)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, l):
        self.l = l

    def __str__(self):
        return "Exp({:.2g})".format(self.l)

    def summary(self):
        ret = {"E[x]": 1. / self.l,
               "E[ln x]": np.nan,
               "var[x]": 1. / self.l**2,
               "Entropy": 1. - np.log(self.l),
               "Mode": 0.}
        return ret

    def lnpdf(self, x):
        return np.log(self.l) - self.l * x

    def lnpdf_grad(self, x):
        return - self.l

    def rvs(self, n):
        return np.random.exponential(scale=self.l, size=n)

class StudentT(Prior):
    """
    Implementation of the student t probability function, coupled with random variables.

    :param mu: mean
    :param sigma: standard deviation
    :param nu: degrees of freedom

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _REAL
    _instances = []

    def __new__(cls, mu=0, sigma=1, nu=4):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().mu == mu and instance().sigma == sigma and instance().nu == nu:
                    return instance()
        newfunc = super(Prior, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)
        else:
            o = newfunc(cls, mu, sigma, nu)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu, sigma, nu):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.sigma2 = np.square(self.sigma)
        self.nu = float(nu)

    def __str__(self):
        return "St({:.2g}, {:.2g}, {:.2g})".format(self.mu, self.sigma, self.nu)

    def lnpdf(self, x):
        from scipy.stats import t
        return t.logpdf(x,self.nu,self.mu,self.sigma)

    def lnpdf_grad(self, x):
        return -(self.nu + 1.)*(x - self.mu)/( self.nu*self.sigma2 + np.square(x - self.mu) )

    def rvs(self, n):
        from scipy.stats import t
        ret = t.rvs(self.nu, loc=self.mu, scale=self.sigma, size=n)
        return ret

