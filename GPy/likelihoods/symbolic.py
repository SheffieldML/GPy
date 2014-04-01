# Copyright (c) 2014 GPy Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

try:
    import sympy as sym
    sympy_available=True
    from sympy.utilities.lambdify import lambdify
except ImportError:
    sympy_available=False

import numpy as np
import link_functions
from scipy import stats, integrate
from scipy.special import gammaln, gamma, erf, polygamma
from GPy.util.functions import cum_gaussian, ln_cum_gaussian
from likelihood import Likelihood
from ..core.parameterization import Param

func_modules = ['numpy', {'gamma':gamma, 'gammaln':gammaln, 'erf':erf,'polygamma':polygamma, 'cum_gaussian':cum_gaussian, 'ln_cum_gaussian':ln_cum_gaussian}]

if sympy_available:
    class Symbolic(Likelihood):
        """
        Symbolic likelihood.

        Likelihood where the form of the likelihood is provided by a sympy expression.

        """
        def __init__(self, pdf=None, log_pdf=None, cdf=None, logZ=None, gp_link=None, name='symbolic', log_concave=False, param=None):
            if gp_link is None:
                gp_link = link_functions.Identity()

            if pdf is None and log_pdf is None and cdf is None:
                raise ValueError, "You must provide an argument for the pdf or the log pdf."

            super(Symbolic, self).__init__(gp_link, name=name)

            if pdf is None and log_pdf:
                self._sp_pdf = sym.exp(log_pdf).simplify()
                self._sp_log_pdf = log_pdf

            if log_pdf is None and pdf:
                self._sp_pdf = pdf
                self._sp_log_pdf = sym.log(pdf).simplify()

            # TODO: build pdf and log pdf from CDF or
            # compute CDF given pdf/log-pdf. Also check log
            # pdf, pdf and CDF are consistent.

            # pull the variable names out of the symbolic pdf
            sp_vars = [e for e in self._sp_pdf.atoms() if e.is_Symbol]
            self._sp_f = [e for e in sp_vars if e.name=='f']
            if not self._sp_f:
                raise ValueError('No variable f in pdf or log pdf.')
            self._sp_y = [e for e in sp_vars if e.name=='y']
            if not self._sp_f:
                raise ValueError('No variable y in pdf or log pdf.')
            self._sp_theta = sorted([e for e in sp_vars if not (e.name=='f' or e.name=='y')],key=lambda e:e.name)

            # These are all the arguments need to compute likelihoods.
            self.arg_list = self._sp_y + self._sp_f + self._sp_theta

            # these are arguments for computing derivatives.
            derivative_arguments = self._sp_f + self._sp_theta

            # Do symbolic work to compute derivatives.
            self._log_pdf_derivatives = {theta.name : sym.diff(self._sp_log_pdf,theta).simplify() for theta in derivative_arguments}
            self._log_pdf_second_derivatives = {theta.name : sym.diff(self._log_pdf_derivatives['f'],theta).simplify() for theta in derivative_arguments}
            self._log_pdf_third_derivatives = {theta.name : sym.diff(self._log_pdf_second_derivatives['f'],theta).simplify() for theta in derivative_arguments}

            # Add parameters to the model.
            for theta in self._sp_theta:
                val = 1.0
                # TODO: need to decide how to handle user passing values for the se parameter vectors.
                if param is not None:
                    if param.has_key(theta):
                        val = param[theta]
                setattr(self, theta.name, Param(theta.name, val, None))
                self.add_parameters(getattr(self, theta.name))


            # Is there some way to check whether the pdf is log
            # concave? For the moment, need user to specify.
            self.log_concave = log_concave

            # initialise code arguments
            self._arguments = {} 

            # generate the code for the pdf and derivatives
            self._gen_code()

        def _gen_code(self):
            """Generate the code from the symbolic parts that will be used for likleihod computation."""
            # TODO: Check here whether theano is available and set up
            # functions accordingly.
            self._pdf_function = lambdify(self.arg_list, self._sp_pdf, func_modules)
            self._log_pdf_function = lambdify(self.arg_list, self._sp_log_pdf, func_modules)

            # compute code for derivatives (for implicit likelihood terms
            # we need up to 3rd derivatives)
            setattr(self, '_first_derivative_code', {key: lambdify(self.arg_list, self._log_pdf_derivatives[key], func_modules) for key in self._log_pdf_derivatives.keys()})
            setattr(self, '_second_derivative_code', {key: lambdify(self.arg_list, self._log_pdf_second_derivatives[key], func_modules) for key in self._log_pdf_second_derivatives.keys()})
            setattr(self, '_third_derivative_code', {key: lambdify(self.arg_list, self._log_pdf_third_derivatives[key], func_modules) for key in self._log_pdf_third_derivatives.keys()})

            # TODO: compute EP code parts based on logZ. We need dlogZ/dmu, d2logZ/dmu2 and dlogZ/dtheta

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
            for grad, theta in zip(grads, self._sp_theta):
                parameter = getattr(self, theta.name)
                setattr(parameter, 'gradient', grad)

        def _arguments_update(self, f, y):
            """Set up argument lists for the derivatives."""
            # If we do make use of Theano, then at this point we would
            # need to do a lot of precomputation to ensure that the
            # likelihoods and gradients are computed together, then check
            # for parameter changes before updating.
            for i, fvar in enumerate(self._sp_f):
                self._arguments[fvar.name] =  f
            for i, yvar in enumerate(self._sp_y):
                self._arguments[yvar.name] = y
            for theta in self._sp_theta:
                self._arguments[theta.name] = np.asarray(getattr(self, theta.name))

        def pdf_link(self, inv_link_f, y, Y_metadata=None):
            """
            Likelihood function given inverse link of f.

            :param inv_link_f: inverse link of latent variables.
            :type inv_link_f: Nx1 array
            :param y: data
            :type y: Nx1 array
            :param Y_metadata: Y_metadata which is not used in student t distribution
            :returns: likelihood evaluated for this point
            :rtype: float
            """
            assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape
            self._arguments_update(inv_link_f, y)
            l = self._pdf_function(**self._arguments)
            return np.prod(l)

        def logpdf_link(self, inv_link_f, y, Y_metadata=None):
            """
            Log Likelihood Function given inverse link of latent variables.

            :param inv_inv_link_f: latent variables (inverse link of f)
            :type inv_inv_link_f: Nx1 array
            :param y: data
            :type y: Nx1 array
            :param Y_metadata: Y_metadata 
            :returns: likelihood evaluated for this point
            :rtype: float

            """
            assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape
            self._arguments_update(inv_link_f, y)
            ll = self._log_pdf_function(**self._arguments)
            return np.sum(ll)

        def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
            """
            Gradient of log likelihood with respect to the inverse link function.

            :param inv_inv_link_f: latent variables (inverse link of f)
            :type inv_inv_link_f: Nx1 array
            :param y: data
            :type y: Nx1 array
            :param Y_metadata: Y_metadata 
            :returns: gradient of likelihood with respect to each point.
            :rtype: Nx1 array

            """
            assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape 
            self._arguments_update(inv_link_f, y)
            return self._first_derivative_code['f'](**self._arguments)

        def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
            """
            Hessian of log likelihood given inverse link of latent variables with respect to that inverse link.
            i.e. second derivative logpdf at y given inv_link(f_i) and inv_link(f_j)  w.r.t inv_link(f_i) and inv_link(f_j).


            :param inv_link_f: inverse link of the latent variables.
            :type inv_link_f: Nx1 array
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
            assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape 
            self._arguments_update(inv_link_f, y)
            return self._second_derivative_code['f'](**self._arguments)

        def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
            assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape 
            self._arguments_update(inv_link_f, y)
            return self._third_derivative_code['f'](**self._arguments)
            raise NotImplementedError

        def dlogpdf_link_dtheta(self, inv_link_f, y, Y_metadata=None):
            assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape 
            self._arguments_update(inv_link_f, y)
            g = np.zeros((y.shape[0], len(self._sp_theta)))
            for i, theta in enumerate(self._sp_theta):
                g[:, i:i+1] = self._first_derivative_code[theta.name](**self._arguments)
            return g.sum(0)

        def dlogpdf_dlink_dtheta(self, inv_link_f, y, Y_metadata=None):
            assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape 
            self._arguments_update(inv_link_f, y)
            g = np.zeros((y.shape[0], len(self._sp_theta)))
            for i, theta in enumerate(self._sp_theta):
                g[:, i:i+1] = self._second_derivative_code[theta.name](**self._arguments)
            return g

        def d2logpdf_dlink2_dtheta(self, inv_link_f, y, Y_metadata=None):
            assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape 
            self._arguments_update(inv_link_f, y)
            g = np.zeros((y.shape[0], len(self._sp_theta)))
            for i, theta in enumerate(self._sp_theta):
                g[:, i:i+1] = self._third_derivative_code[theta.name](**self._arguments)
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
