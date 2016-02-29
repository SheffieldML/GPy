# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 20:27:30 2015

@author: Felix
"""
from ..inference import optimization
from ..core.parameterization import Param
from ..core.model import Model
import numpy as np
import logging
from numpy.linalg.linalg import LinAlgError
import math


logger = logging.getLogger("GPHeteroscedasticRegression")


class GPHeteroscedasticRegression(Model):
    """
    Gaussian Process model for heteroscedastic regression

    :param X: input observations
    :param Y: observed values
    Other parameters are there to give it the same interface as other Models but are not used
    This model uses its own Square Exponentional kernel
    """
    def __init__(self, X, Y, kernel=None, varKernel = None, optimize = True):
        """
        Initializes all values and links the parameters which is needed for the optimizer array
        """
        super(GPHeteroscedasticRegression, self).__init__("GPHeteroscedasticRegression")
        self.Y = Y
        self.X = X
        _, self.output_dim = self.Y.shape

        self.num_data, self.input_dim = self.X.shape
        self.kernel  =  SEKernel(self.input_dim)
        self.varKernel  = SEKernel(self.input_dim)
        self.mean = 1

        self.n,self.D = X.shape
        self.Lambda0 = np.ones([self.n,1])*math.log(0.5);
        self.theta1 = np.ones([self.input_dim+1])
        self.theta2 = np.ones([self.input_dim+1])

        self.Lambda0 = Param('Lambda0', self.Lambda0)
        self.theta1 = Param('theta1', self.theta1 )
        self.theta2 = Param('theta2', self.theta2 )
        self.mean = Param('mean', self.mean)

        print("link hyperparameters")
        self.link_parameter(self.Lambda0,index=0)
        self.link_parameter(self.theta1,index=1)
        self.link_parameter(self.theta2, index = 2)
        self.link_parameter(self.mean, index = 3)

    def train(self, X, Y):
        """
        Takes new X and Y to train.
        """
        self.X = X
        self.Y = Y
        if X.size == 0 or Y.size == 0:
            return
        self.n,self.D = X.shape
        self.Lambda0 = np.ones([self.n,1])*math.log(0.5);
        self.Lambda0 = Param('Lambda0', self.Lambda0)
        self.link_parameter(self.Lambda0,index=0)
        if self.optimize:
            self.optimize_restarts(num_restarts=self.num_restarts, robust=True)

    def optimize(self):
        """
        Optimizes by calling optimize_without_kernels and optimize_with_kernels with default steps in each.
        """
        self.optimize_without_kernels();
        self.optimize_with_kernels();

    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method reperforms inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.setup()
        self.derivatives(True,True)

    def setup(self,verbose =0):
        if verbose > 0:
            print("Setup and calculate objective function")
        self.Lambda = np.exp(self.Lambda0);
        self.Kf = self.kernel.CovarianceMatrix(self.theta1,self.X)
        self.Kg = self.varKernel.CovarianceMatrix(self.theta2,self.X)
        sLambda = np.sqrt(self.Lambda);
        self.cinvB = np.linalg.solve(np.transpose(np.linalg.cholesky(np.eye(self.n) + self.Kg*np.dot(sLambda,np.transpose(sLambda))).T),np.eye(self.n));   # O(n^3)
        self.cinvBs = self.cinvB*(np.dot(np.ones([self.n,1]),sLambda.T));
        self.beta = (self.Lambda-0.5)
        mu = np.dot(self.Kg,self.beta)+self.mean
        
        hBLK2 = np.dot(self.cinvBs,self.Kg);
        self.Sigma = self.Kg - np.dot(np.transpose(hBLK2),hBLK2);
        
        R = np.exp(mu-np.transpose(np.array([np.diag(self.Sigma)/2])));
        p = 1e-3; # This value doesn't affect the result, it is a scaling factor
        self.scale = np.divide(1,np.sqrt(p+R)); 
        self.Rscale = np.divide(1,(1+np.divide(p,R)));
        self.Ls = np.linalg.cholesky(self.Kf*(np.dot(self.scale,np.transpose(self.scale)))+np.diagflat(self.Rscale)).T;
        Lys = np.linalg.solve(np.transpose(self.Ls),(self.Y*self.scale));
        self.alphascale = np.linalg.solve(self.Ls,Lys);
        self.alpha = self.alphascale*self.scale;
        
        # --- Objective
        F = (-0.5*np.dot(np.transpose(self.Y),self.alpha) - np.sum(np.log(np.diag(self.Ls)))  + np.sum(np.log(self.scale))  - self.n/2.0*np.log(2*np.pi)  # log N(y|0,Kf+R)
            -0.5*np.dot(np.transpose(self.beta),(mu-self.mean)) + np.sum(np.log(np.diag(self.cinvB))) - 0.5*np.sum(np.sum(np.square(self.cinvB))) + self.n/2.0            # -KL(N(g|mu,Sigma)||N(g|0,Kg))
            -0.25*np.trace(self.Sigma));                                                                      # Normalization
        
        self.obj_function=-F;

    def derivatives(self, optimizeKernel = False, optimizeVarKernel = False, verbose = 0):
        if verbose > 0:
            print("Caluclate Derivates")

        # --- Derivatives
        self.derivate_obj_func=np.zeros([self.n,1]);
        self.derivate_mean_kernel_obj_func = np.zeros([self.kernel.input_dim+1,1]);
        self.derivate_var_kernel_obj_func = np.zeros([self.varKernel.input_dim+1,1]);

        invKfRs = np.linalg.solve(self.Ls,(np.linalg.solve(self.Ls.T,np.eye(self.n))));    # O(n^3)
        betahat = -0.5*(np.array([np.diag(invKfRs)]).T*self.Rscale-np.square(self.alphascale)*self.Rscale);
        Lambdahat = betahat + 0.5;
        
        # wrt Lambda
        dFLambda = -np.dot(self.Kg+np.dot(0.5,np.square(self.Sigma)),(self.Lambda-Lambdahat));
        self.derivate_obj_func[0:self.n] = -self.Lambda*dFLambda;
        
        if optimizeKernel:
            # wrt Kf hyperparameters
            W = np.dot(self.alpha,np.transpose(self.alpha))- invKfRs*(np.dot(self.scale,np.transpose(self.scale)));
            for k in range(0,self.kernel.input_dim+1):
                self.derivate_mean_kernel_obj_func[k] = -np.sum(np.sum(W*self.kernel.computeDerivateMatrixWithRespectToZ(self.theta1,self.X, k)))/2;
        if optimizeVarKernel:
            # wrt Kg hyperparameters
            invBs = np.dot(np.transpose(self.cinvB),self.cinvBs);    # O(n^3)
            W = (np.dot(self.beta,self.beta.T) + 2*np.dot((betahat-self.beta),self.beta.T)
            - np.dot(invBs.T,(np.dot(np.divide(Lambdahat,self.Lambda)-1,np.ones([1,self.n]))*invBs))
            - np.dot(self.cinvBs.T,self.cinvBs));
            for k in range(0,self.varKernel.input_dim+1):
                self.derivate_var_kernel_obj_func[k] = -np.sum(np.sum(W*self.varKernel.computeDerivateMatrixWithRespectToZ(self.theta2,self.X, k)))/2;
                
        # wrt self.mean
        dFmu0 = sum(Lambdahat-0.5);
        self.derivate_mean = -dFmu0;
        
    def predictive_gradients(self,Xnew):
        return NotImplementedError

    def plot(self,*args):
        return NotImplementedError

    def optimize_without_kernels(self, optimizer=None, start=None, **kwargs):
        """
        Optimize the model without the kernels

        kwargs are passed to the optimizer. They can be:

        :param max_f_eval: maximum number of function evaluations
        :type max_f_eval: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred optimizer)
        :type optimizer: string

        Valid optimizers are:
          - 'scg': scaled conjugate gradient method, recommended for stability.
                   See also GPy.inference.optimization.scg
          - 'fmin_tnc': truncated Newton method (see scipy.optimize.fmin_tnc)
          - 'simplex': the Nelder-Mead simplex method (see scipy.optimize.fmin),
          - 'lbfgsb': the l-bfgs-b method (see scipy.optimize.fmin_l_bfgs_b),
          - 'sgd': stochastic gradient decsent (see scipy.optimize.sgd). For experts only!


        """
        if self.is_fixed or self.size == 0:
            print('nothing to optimize')

        if not self.update_model():
            print("updates were off, setting updates on again")
            self.update_model(True)

        if start == None:
            start = np.hstack([self.Lambda0.flatten(),self.theta1.flatten()])
            start = np.hstack([start,self.theta2.flatten()])
            start = np.hstack([start,self.mean])
        if optimizer is None:
            optimizer = self.preferred_optimizer

        if isinstance(optimizer, optimization.Optimizer):
            opt = optimizer
            opt.model = self
        else:
            optimizer = optimization.get_optimizer(optimizer)
            opt = optimizer(start, model=self,messages=True, **kwargs)

        opt.run(f_fp=self._objective_grads_without_kernels, f=self._objective, fp=self._grads_without_kernels)

        self.optimization_runs.append(opt)

        self.optimizer_array = opt.x_opt

    def optimize_with_kernels(self, optimizer=None, start=None, **kwargs):
        """
        Optimize the model with the kernels

        kwargs are passed to the optimizer. They can be:

        :param max_f_eval: maximum number of function evaluations
        :type max_f_eval: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred optimizer)
        :type optimizer: string

        Valid optimizers are:
          - 'scg': scaled conjugate gradient method, recommended for stability.
                   See also GPy.inference.optimization.scg
          - 'fmin_tnc': truncated Newton method (see scipy.optimize.fmin_tnc)
          - 'simplex': the Nelder-Mead simplex method (see scipy.optimize.fmin),
          - 'lbfgsb': the l-bfgs-b method (see scipy.optimize.fmin_l_bfgs_b),
          - 'sgd': stochastic gradient decsent (see scipy.optimize.sgd). For experts only!


        """
        if self.is_fixed:
            print 'nothing to optimize'
        if self.size == 0:
            print 'nothing to optimize'

        if not self.update_model():
            print "setting updates on again"
            self.update_model(True)

        if start == None:
            start = np.hstack([self.Lambda0.flatten(),self.theta1.flatten()])
            start = np.hstack([start,self.theta2.flatten()])
            start = np.hstack([start,self.mean])

        if optimizer is None:
            optimizer = self.preferred_optimizer

        if isinstance(optimizer, optimization.Optimizer):
            opt = optimizer
            opt.model = self
        else:
            optimizer = optimization.get_optimizer(optimizer)
            opt = optimizer(start, model=self, **kwargs)

        opt.run(f_fp=self._objective_grads_with_kernels, f=self._objective, fp=self._grads_with_kernels)

        self.optimization_runs.append(opt)

        self.optimizer_array = opt.x_opt

    def objective_function(self):
        """
        The objective function for the given algorithm.
        """
        return self.obj_function

    def objective_function_gradients_without_kernels(self):
        """
        The gradients for the objective function without kernels
        This function is the true objective, which wants to be minimized in case of training without kernels.
        """
        allDerivates = np.append(self.derivate_obj_func,np.zeros([self.input_dim+1,1]))
        allDerivates = np.append(allDerivates, np.zeros([self.input_dim+1,1]))
        allDerivates = np.append(allDerivates, self.derivate_mean)
        return allDerivates

    def objective_function_gradients_with_kernels(self):
        """
        The gradients for the objective function with kernels
        This function is the true objective, which wants to be minimized in case of training with kernels.
        """
        allDerivates = np.append(self.derivate_obj_func,self.derivate_mean_kernel_obj_func)
        allDerivates = np.append(allDerivates, self.derivate_var_kernel_obj_func)
        allDerivates = np.append(allDerivates, self.derivate_mean)
        return allDerivates

    def _objective_grads_without_kernels(self, x):
        try:
            self.optimizer_array = x
            obj_f, obj_grads = self.objective_function(), self._transform_gradients(self.objective_function_gradients_without_kernels())
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError):
            if self._fail_count >= self._allowed_failures:
                raise
            self._fail_count += 1
            obj_f = np.inf
            obj_grads = np.clip(self._transform_gradients(self.objective_function_gradients_without_kernels()), -1e100, 1e100)
        return obj_f, obj_grads

    def _objective_grads_with_kernels(self, x):
        try:
            self.optimizer_array = x
            obj_f, obj_grads = self.objective_function(), self._transform_gradients(self.objective_function_gradients_with_kernels())
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError):
            if self._fail_count >= self._allowed_failures:
                raise
            self._fail_count += 1
            obj_f = np.inf
            obj_grads = np.clip(self._transform_gradients(self.objective_function_gradients_with_kernels()), -1e100, 1e100)
        return obj_f, obj_grads

    def _grads_without_kernels(self, x):
        """
        Gets the gradients from the likelihood and the priors without kernels

        Failures are handled robustly. The algorithm will try several times to
        return the gradients, and will raise the original exception if
        the objective cannot be computed.

        :param x: the parameters of the model.
        :type x: np.array
        """
        try:
            self.optimizer_array = x
            self.obj_grads_without_kernels = self._transform_gradients(self.objective_function_gradients_without_kernels())
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError):
            if self._fail_count >= self._allowed_failures:
                raise
            self._fail_count += 1
            self.obj_grads_without_kernels = np.clip(self._transform_gradients(self.objective_function_gradients_without_kernels()), -1e100, 1e100)
        return self.obj_grads_without_kernels

    def _grads_with_kernels(self, x):
        """
        Gets the gradients from the likelihood and the priors with kernels

        Failures are handled robustly. The algorithm will try several times to
        return the gradients, and will raise the original exception if
        the objective cannot be computed.

        :param x: the parameters of the model.
        :type x: np.array
        """
        try:
            self.optimizer_array = x
            self.obj_grads_with_kernels = self._transform_gradients(self.objective_function_gradients_with_kernels())
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError):
            if self._fail_count >= self._allowed_failures:
                raise
            self._fail_count += 1
            self.obj_grads_with_kernels = np.clip(self._transform_gradients(self.objective_function_gradients_with_kernels()), -1e100, 1e100)
        return self.obj_grads_with_kernels

    def predict(self,X,verbose = 0):
        if(verbose>0):
            print("predictPoints")
        [K1ss, K1star] = self.kernel.TestSetCovariances(self.theta1,self.X, X);         # test covariance f
        [K2ss, K2star] = self.varKernel.TestSetCovariances(self.theta2,self.X, X);      # test covariance g
        atst  = np.dot(np.transpose(K1star), self.alpha);                               # predicted mean  f
        mutst = np.dot(np.transpose(K2star), self.beta) + self.mean;                    # predicted mean  g
        out1 = atst;                                                                    # predicted mean  y

        v = np.linalg.solve(np.transpose(self.Ls),((np.dot(self.scale,np.ones([1,np.size(X,0)])))*K1star));
        if K1ss.shape[1]== 1:
            diagCtst = K1ss - np.array([np.sum(v*v,0)]).T;
        else:
            diagCtst = K1ss - np.transpose(np.sum(v*v,0));                              # predicted variance f
        v = np.dot(self.cinvBs,K2star);
        if K2ss.shape[1]== 1:
            diagSigmatst = K2ss - np.array([np.sum(v*v,0)]).T;                          # predicted variance g
        else:
            diagSigmatst = K2ss - np.transpose(np.sum(v*v,0));                          # predicted variance g
        out2 = diagCtst + np.exp(mutst+diagSigmatst/2);                                 # predicted variance y
        return [out1,out2]






class SEKernel():
    """
    Square exponential kernel needed to run heteroscedastic_regression
    """
    def __init__(self, input_dim):

        self.input_dim = input_dim
        self.jitter = 1e-6;

    def numOfParams(self):
        return self.input_dim+1

    def CovarianceMatrix(self,hyperparameters,X):
        [n, D] = X.shape;
        ell = np.exp(hyperparameters[0:D]);        # characteristic length scale
        sf2 = np.exp(2*hyperparameters[D]);      # signal variance
        self.K = sf2*np.exp(-self.sq_dist(np.dot(np.diagflat(1./ell),X.T))/2);
        A = self.K+sf2*self.jitter*np.eye(n);
        return A

    def TestSetCovariances(self,hyperparameters,X,Y):                   #compute test set covariances
        [n, D] = X.shape;
        ell = np.exp(hyperparameters[0:D]);        # characteristic length scale
        sf2 = np.exp(2*hyperparameters[D]);      # signal variance
        A = np.dot(sf2*(1+self.jitter),np.ones([np.size(Y,0),1]));
        B = sf2*np.exp(-self.sq_dist2(np.dot(np.diagflat(1./ell),X.T),np.dot(np.diagflat(1./ell),Y.T))/2);#TODO diagflat
        return [A,B] #TODO [A,B]

    def computeDerivateMatrixWithRespectToZ(self,hyperparameters,X,z):
        [n, D] = X.shape;
        ell = np.exp(hyperparameters[0:D]);        # characteristic length scale
        sf2 = np.exp(2*hyperparameters[D]);      # signal variance
        if z == 0:
            self.K = np.dot(sf2,np.exp(-self.sq_dist(np.dot(np.diagflat(1./ell),np.transpose(X)))/2));
        if z < D:                                           # length scale parameters
            if D == 1 :
                A = self.K*self.sq_dist(np.array([X[:,z]/ell]));
            else:
                A = self.K*self.sq_dist(np.array([np.transpose(X[:,z])/ell[z]]));
        else :                                                    # magnitude parameter
            A = 2*(self.K+np.dot(sf2,self.jitter*np.eye(n)));
            self.K = None
        return A



    def sq_dist(self,a):
        [D, n] = a.shape;
        mu = np.mean(a);
        a = a - np.tile(mu,(1,a.shape[1]))
        b = a
        m = n
        C = np.tile(np.array([np.sum (a*a,axis = 0)]).T,(1,m)) + np.tile(np.sum(b*b,axis = 0),(n,1)) - np.dot(2*a.T,b);
        C = np.maximum(C,0);          # numerical noise can cause C to negative i.e. C > -1e-14
        return C


    def sq_dist2(self,a,b):
        [D, n] = a.shape;
        [d, m] = b.shape;
        mu = (np.array([(m/(n+m+0.0))*np.mean(b,1) + (n/(n+m+0.0))*np.mean(a,1)])).T;
        a = a -np.tile(mu,(1,n))
        b = b -np.tile(mu,(1,m))
        C = np.tile(np.array([np.sum (a*a,axis = 0)]).T,(1,m)) + np.tile(np.sum(b*b,axis = 0),(n,1)) - np.dot(2*a.T,b);
        C = np.maximum(C,0);          # numerical noise can cause C to negative i.e. C > -1e-14
        return C