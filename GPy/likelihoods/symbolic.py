# Copyright (c) 2014 GPy Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import sympy as sym
import numpy as np
from likelihood import Likelihood
from ..core.symbolic import Symbolic_core


class Symbolic(Likelihood, Symbolic_core):
    """
    Symbolic likelihood.

    Likelihood where the form of the likelihood is provided by a sympy expression.

    """
    def __init__(self, log_pdf=None, logZ=None, missing_log_pdf=None, gp_link=None, name='symbolic', log_concave=False, parameters=None, func_modules=[]):

        if gp_link is None:
            gp_link = link_functions.Identity()

        if log_pdf is None:
            raise ValueError, "You must provide an argument for the log pdf."

        Likelihood.__init__(self, gp_link, name=name)
        functions = {'log_pdf':log_pdf}
        self.cacheable = ['F', 'Y']

        self.missing_data = False
        if missing_log_pdf:
            self.missing_data = True
            functions['missing_log_pdf']=missing_log_pdf

        self.ep_analytic = False
        if logZ:
            self.ep_analytic = True
            functions['logZ'] = logZ
            self.cacheable += ['M', 'V']            

        Symbolic_core.__init__(self, functions, cacheable=self.cacheable, derivatives = ['F', 'theta'], parameters=parameters, func_modules=func_modules)   

        # TODO: Is there an easy way to check whether the pdf is log
        self.log_concave = log_concave

     

    def _set_derivatives(self, derivatives):
        # these are arguments for computing derivatives.
        print "Whoop"
        Symbolic_core._set_derivatives(self, derivatives)

        # add second and third derivatives for Laplace approximation.
        derivative_arguments = []
        if derivatives is not None:
            for derivative in derivatives:
                derivative_arguments += self.variables[derivative]
            exprs = ['log_pdf']
            if self.missing_data:
                exprs.append('missing_log_pdf')
            for expr in exprs:
                self.expressions[expr]['second_derivative'] = {theta.name : self.stabilize(sym.diff(self.expressions[expr]['derivative']['f_0'], theta)) for theta in derivative_arguments}
                self.expressions[expr]['third_derivative'] = {theta.name : self.stabilize(sym.diff(self.expressions[expr]['second_derivative']['f_0'], theta)) for theta in derivative_arguments}
        if self.ep_analytic:
            derivative_arguments = [M]
            # add second derivative for EP 
            exprs = ['logZ']
            if self.missing_data:
                exprs.append('missing_logZ')
            for expr in exprs:
                self.expressions[expr]['second_derivative'] = {theta.name : self.stabilize(sym.diff(self.expressions[expr]['derivative'], theta)) for theta in derivative_arguments}


    def eval_update_cache(self, Y, **kwargs):
        # TODO: place checks for inf/nan in here
        # for all provided keywords
        Symbolic_core.eval_update_cache(self, Y=Y, **kwargs)
        # Y = np.atleast_2d(Y)
        # for variable, code in sorted(self.code['parameters_changed'].iteritems()):
        #     self._set_attribute(variable, eval(code, self.namespace))
        # for i, theta in enumerate(self.variables['Y']):
        #     missing = np.isnan(Y[:, i])
        #     self._set_attribute('missing_' + str(i), missing)
        #     self._set_attribute(theta.name, value[missing, i][:, None])
        # for variable, value in kwargs.items():
        #     # update their cached values
        #     if value is not None:
        #         if variable == 'F' or variable == 'M' or variable == 'V' or variable == 'Y_metadata':
        #             for i, theta in enumerate(self.variables[variable]):
        #                 self._set_attribute(theta.name, value[:, i][:, None])
        #         else:
        #             self._set_attribute(theta.name, value[:, i])
        # for variable, code in sorted(self.code['update_cache'].iteritems()):
        #     self._set_attribute(variable, eval(code, self.namespace))


    def parameters_changed(self):
        pass

    def update_gradients(self, grads):
        """
        Pull out the gradients, be careful as the order must match the order
        in which the parameters are added
        """
        # The way the Laplace approximation is run requires the
        # covariance function to compute the true gradient (because it
        # is dependent on the mode). This means we actually compute
        # the gradient outside this object. This function would
        # normally ask the object to update its gradients internally,
        # but here it provides them externally, because they are
        # computed in the inference code. TODO: Thought: How does this
        # effect EP? Shouldn't this be done by a separate
        # Laplace-approximation specific call?
        for theta, grad in zip(self.variables['theta'], grads):
            parameter = getattr(self, theta.name)
            setattr(parameter, 'gradient', grad)

    def pdf_link(self, f, y, Y_metadata=None):
        """
        Likelihood function given inverse link of f.

        :param f: inverse link of latent variables.
        :type f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in student t distribution
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        return np.exp(self.logpdf_link(f, y, Y_metadata=None))

    def logpdf_link(self, f, y, Y_metadata=None):
        """
        Log Likelihood Function given inverse link of latent variables.

        :param f: latent variables (inverse link of f)
        :type f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata 
        :returns: likelihood evaluated for this point
        :rtype: float

        """
        assert np.atleast_1d(f).shape == np.atleast_1d(y).shape
        if self.missing_data:
            missing_flag = np.isnan(y)
            not_missing_flag = np.logical_not(missing_flag)
            ll = self.eval_function('missing_log_pdf', F=f[missing_flag]).sum()
            ll += self.eval_function('log_pdf', F=f[not_missing_flag], Y=y[not_missing_flag], Y_metadata=Y_metadata[not_missing_flag]).sum()
        else:
            ll = self.eval_function('log_pdf', F=f, Y=y, Y_metadata=Y_metadata).sum()

        return ll

    def dlogpdf_dlink(self, f, y, Y_metadata=None):
        """
        Gradient of log likelihood with respect to the inverse link function.

        :param f: latent variables (inverse link of f)
        :type f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata 
        :returns: gradient of likelihood with respect to each point.
        :rtype: Nx1 array

        """
        assert np.atleast_1d(f).shape == np.atleast_1d(y).shape 
        self.eval_update_cache(F=f, Y=y, Y_metadata=Y_metadata)
        if self.missing_data:
            return np.where(np.isnan(y), 
                            eval(self.code['missing_log_pdf']['derivative']['f_0'], self.namespace), 
                            eval(self.code['log_pdf']['derivative']['f_0'], self.namespace)) 
        else:
            return np.where(np.isnan(y), 
                            0., 
                            eval(self.code['log_pdf']['derivative']['f_0'], self.namespace))

    def d2logpdf_dlink2(self, f, y, Y_metadata=None):
        """
        Hessian of log likelihood given inverse link of latent variables with respect to that inverse link.
        i.e. second derivative logpdf at y given inv_link(f_i) and inv_link(f_j)  w.r.t inv_link(f_i) and inv_link(f_j).


        :param f: inverse link of the latent variables.
        :type f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in student t distribution
        :returns: Diagonal of Hessian matrix (second derivative of likelihood evaluated at points f)
        :rtype: Nx1 array

        .. Note::
            Returns diagonal of Hessian, since every where else it is
            0, as the likelihood factorizes over cases (the
            distribution for y_i depends only on link(f_i) not on
            link(f_(j!=i))
        """
        assert np.atleast_1d(f).shape == np.atleast_1d(y).shape         
        self.eval_update_cache(F=f, Y=y, Y_metadata=Y_metadata)
        if self.missing_data:
            return np.where(np.isnan(y), 
                            eval(self.code['missing_log_pdf']['second_derivative']['f_0'], self.namespace), 
                            eval(self.code['log_pdf']['second_derivative']['f_0'], self.namespace)) 
        else:
            return np.where(np.isnan(y), 
                            0., 
                            eval(self.code['log_pdf']['second_derivative']['f_0'], self.namespace))

    def d3logpdf_dlink3(self, f, y, Y_metadata=None):
        assert np.atleast_1d(f).shape == np.atleast_1d(y).shape 
        self.eval_update_cache(F=f, Y=y, Y_metadata=Y_metadata)
        if self.missing_data:
            return np.where(np.isnan(y), 
                            eval(self.code['missing_log_pdf']['third_derivative']['f_0'], self.namespace), 
                            eval(self.code['log_pdf']['third_derivative']['f_0'], self.namespace))
        else:
            return np.where(np.isnan(y), 
                            0., 
                            eval(self.code['log_pdf']['third_derivative']['f_0'], self.namespace))

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        assert np.atleast_1d(f).shape == np.atleast_1d(y).shape 
        self.eval_update_cache(F=f, Y=y, Y_metadata=Y_metadata)
        g = np.zeros((np.atleast_1d(y).shape[0], len(self.variables['theta'])))
        for i, theta in enumerate(self.variables['theta']):
            if self.missing_data:
                g[:, i:i+1] = np.where(np.isnan(y), 
                                       eval(self.code['missing_log_pdf']['derivative'][theta.name], self.namespace), 
                                       eval(self.code['log_pdf']['derivative'][theta.name], self.namespace))
            else:
                g[:, i:i+1] = np.where(np.isnan(y), 
                                       0., 
                                       eval(self.code['log_pdf']['derivative'][theta.name], self.namespace))
        return g.sum(0)

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        assert np.atleast_1d(f).shape == np.atleast_1d(y).shape 
        self.eval_update_cache(F=f, Y=y, Y_metadata=Y_metadata)
        g = np.zeros((np.atleast_1d(y).shape[0], len(self.variables['theta'])))
        for i, theta in enumerate(self.variables['theta']):
            if self.missing_data:
                g[:, i:i+1] = np.where(np.isnan(y), 
                                       eval(self.code['missing_log_pdf']['second_derivative'][theta.name], self.namespace), 
                                       eval(self.code['log_pdf']['second_derivative'][theta.name], self.namespace))
            else:
                g[:, i:i+1] = np.where(np.isnan(y), 
                                       0., 
                                       eval(self.code['log_pdf']['second_derivative'][theta.name], self.namespace))
        return g

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        assert np.atleast_1d(f).shape == np.atleast_1d(y).shape 
        self.eval_update_cache(F=f, Y=y, Y_metadata=Y_metadata)
        g = np.zeros((np.atleast_1d(y).shape[0], len(self.variables['theta'])))
        for i, theta in enumerate(self.variables['theta']):
            if self.missing_data:
                g[:, i:i+1] = np.where(np.isnan(y), 
                                       eval(self.code['missing_log_pdf']['third_derivative'][theta.name], self.namespace), 
                                       eval(self.code['log_pdf']['third_derivative'][theta.name], self.namespace))
            else:
                g[:, i:i+1] = np.where(np.isnan(y), 
                                       0., 
                                       eval(self.code['log_pdf']['third_derivative'][theta.name], self.namespace))
        return g

    def predictive_mean(self, mu, sigma, Y_metadata=None):
        raise NotImplementedError

    def predictive_variance(self, mu,variance, predictive_mean=None, Y_metadata=None):
        raise NotImplementedError

    def conditional_mean(self, gp):
        raise NotImplementedError

    def conditional_variance(self, gp):
        raise NotImplementedError

    def samples(self, gp, Y_metadata=None):
        raise NotImplementedError
