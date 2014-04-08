# Check Matthew Rocklin's blog post.
try:
    import sympy as sym
    sympy_available=True
    from sympy.utilities.lambdify import lambdify
    from GPy.util.symbolic import stabilise
except ImportError:
    sympy_available=False

import numpy as np
from kern import Kern
from scipy.special import gammaln, gamma, erf, erfc, erfcx, polygamma
from GPy.util.functions import normcdf, normcdfln, logistic, logisticln, differfln
from ...core.parameterization import Param

class Symbolic(Kern):
    """
    A kernel object, where all the hard work is done by sympy.

    :param k: the covariance function
    :type k: a positive definite sympy function of x_0, z_0, x_1, z_1, x_2, z_2...

    To construct a new sympy kernel, you'll need to define:
     - a kernel function using a sympy object. Ensure that the kernel is of the form k(x,z).
     - that's it! we'll extract the variables from the function k.

    Note:
     - to handle multiple inputs, call them x_1, z_1, etc
     - to handle multpile correlated outputs, you'll need to add parameters with an index, such as lengthscale_i and lengthscale_j.
    """
    def __init__(self, input_dim, k=None, output_dim=1, name='symbolic', param=None, active_dims=None, operators=None, func_modules=[]):

        if k is None:
            raise ValueError, "You must provide an argument for the covariance function."

        self.func_modules = func_modules
        self.func_modules += [{'gamma':gamma,
                               'gammaln':gammaln,
                               'erf':erf, 'erfc':erfc,
                               'erfcx':erfcx,
                               'polygamma':polygamma,
                               'differfln':differfln,
                               'normcdf':normcdf,
                               'normcdfln':normcdfln,
                               'logistic':logistic,
                               'logisticln':logisticln},
                              'numpy']

        super(Symbolic, self).__init__(input_dim, active_dims, name)

        self._sym_k = k

        # pull the variable names out of the symbolic covariance function.
        sym_vars = [e for e in k.atoms() if e.is_Symbol]
        self._sym_x= sorted([e for e in sym_vars if e.name[0:2]=='x_'],key=lambda x:int(x.name[2:]))
        self._sym_z= sorted([e for e in sym_vars if e.name[0:2]=='z_'],key=lambda z:int(z.name[2:]))

        # Check that variable names make sense.
        assert all([x.name=='x_%i'%i for i,x in enumerate(self._sym_x)])
        assert all([z.name=='z_%i'%i for i,z in enumerate(self._sym_z)])
        assert len(self._sym_x)==len(self._sym_z)
        x_dim=len(self._sym_x)

        self._sym_kdiag = k
        for x, z in zip(self._sym_x, self._sym_z):
            self._sym_kdiag = self._sym_kdiag.subs(z, x)

        # If it is a multi-output covariance, add an input for indexing the outputs.
        self._real_input_dim = x_dim
        # Check input dim is number of xs + 1 if output_dim is >1
        assert self.input_dim == x_dim + int(output_dim > 1)
        self.output_dim = output_dim

        # extract parameter names from the covariance
        thetas = sorted([e for e in sym_vars if not (e.name[0:2]=='x_' or e.name[0:2]=='z_')],key=lambda e:e.name)

        # Look for parameters with index (subscripts), they are associated with different outputs.
        if self.output_dim>1:
            self._sym_theta_i = sorted([e for e in thetas if (e.name[-2:]=='_i')], key=lambda e:e.name)
            self._sym_theta_j = sorted([e for e in thetas if (e.name[-2:]=='_j')], key=lambda e:e.name)

            # Make sure parameter appears with both indices!
            assert len(self._sym_theta_i)==len(self._sym_theta_j)
            assert all([theta_i.name[:-2]==theta_j.name[:-2] for theta_i, theta_j in zip(self._sym_theta_i, self._sym_theta_j)])

            # Extract names of shared parameters (those without a subscript)
            self._sym_theta = [theta for theta in thetas if theta not in self._sym_theta_i and theta not in self._sym_theta_j]

            self.num_split_params = len(self._sym_theta_i)
            self._split_theta_names = ["%s"%theta.name[:-2] for theta in self._sym_theta_i]
            # Add split parameters to the model.
            for theta in self._split_theta_names:
                # TODO: what if user has passed a parameter vector, how should that be stored and interpreted?
                setattr(self, theta, Param(theta, np.ones(self.output_dim), None))
                self.add_parameter(getattr(self, theta))


            self.num_shared_params = len(self._sym_theta)
            for theta_i, theta_j in zip(self._sym_theta_i, self._sym_theta_j):
                self._sym_kdiag = self._sym_kdiag.subs(theta_j, theta_i)

        else:
            self.num_split_params = 0
            self._split_theta_names = []
            self._sym_theta = thetas
            self.num_shared_params = len(self._sym_theta)

        # Add parameters to the model.
        for theta in self._sym_theta:
            val = 1.0
            # TODO: what if user has passed a parameter vector, how should that be stored and interpreted? This is the old way before params class.
            if param is not None:
                if param.has_key(theta.name):
                    val = param[theta.name]
            setattr(self, theta.name, Param(theta.name, val, None))
            self.add_parameters(getattr(self, theta.name))

        # Differentiate with respect to parameters.
        derivative_arguments = self._sym_x + self._sym_theta
        if self.output_dim > 1:
            derivative_arguments += self._sym_theta_i

        self.derivatives = {theta.name : stabilise(sym.diff(self._sym_k,theta)) for theta in derivative_arguments}
        self.diag_derivatives = {theta.name : stabilise(sym.diff(self._sym_kdiag,theta)) for theta in derivative_arguments}

        # This gives the parameters for the arg list.
        self.arg_list = self._sym_x + self._sym_z + self._sym_theta
        self.diag_arg_list = self._sym_x + self._sym_theta
        if self.output_dim > 1:
            self.arg_list += self._sym_theta_i + self._sym_theta_j
            self.diag_arg_list += self._sym_theta_i

        # Check if there are additional linear operators on the covariance.
        self._sym_operators = operators
        # TODO: Deal with linear operators
        #if self._sym_operators:
        #    for operator in self._sym_operators:
                
        # psi_stats aren't yet implemented.
        if False:
            self.compute_psi_stats()

        # generate the code for the covariance functions
        self._gen_code()

    def __add__(self,other):
        return spkern(self._sym_k+other._sym_k)

    def _gen_code(self):
        #fn_theano = theano_function([self.arg_lists], [self._sym_k + self.derivatives], dims={x: 1}, dtypes={x_0: 'float64', z_0: 'float64'})
        self._K_function = lambdify(self.arg_list, self._sym_k, self.func_modules)
        self._K_derivatives_code = {key: lambdify(self.arg_list, self.derivatives[key], self.func_modules) for key in self.derivatives.keys()}
        self._Kdiag_function = lambdify(self.diag_arg_list, self._sym_kdiag, self.func_modules)
        self._Kdiag_derivatives_code = {key: lambdify(self.diag_arg_list, self.diag_derivatives[key], self.func_modules) for key in self.diag_derivatives.keys()}

    def K(self,X,X2=None):
        self._K_computations(X, X2)
        return self._K_function(**self._arguments)


    def Kdiag(self,X):
        self._K_computations(X)
        return self._Kdiag_function(**self._diag_arguments)

    def _param_grad_helper(self,partial,X,Z,target):
        pass


    def gradients_X(self, dL_dK, X, X2=None):
        #if self._X is None or X.base is not self._X.base or X2 is not None:
        self._K_computations(X, X2)
        gradients_X = np.zeros((X.shape[0], X.shape[1]))
        for i, x in enumerate(self._sym_x):
            gf = self._K_derivatives_code[x.name]
            gradients_X[:, i] = (gf(**self._arguments)*dL_dK).sum(1)
        if X2 is None:
            gradients_X *= 2
        return gradients_X

    def gradients_X_diag(self, dL_dK, X):
        self._K_computations(X)
        dX = np.zeros_like(X)
        for i, x in enumerate(self._sym_x):
            gf = self._Kdiag_derivatives_code[x.name]
            dX[:, i] = gf(**self._diag_arguments)*dL_dK
        return dX

    def update_gradients_full(self, dL_dK, X, X2=None):
        # Need to extract parameters to local variables first
        self._K_computations(X, X2)
        for theta in self._sym_theta:
            parameter = getattr(self, theta.name)
            gf = self._K_derivatives_code[theta.name]
            gradient = (gf(**self._arguments)*dL_dK).sum()
            if X2 is not None:
                gradient += (gf(**self._reverse_arguments)*dL_dK).sum()
            setattr(parameter, 'gradient', gradient)
        if self.output_dim>1:
            for theta in self._sym_theta_i:
                parameter = getattr(self, theta.name[:-2])
                gf = self._K_derivatives_code[theta.name]
                A = gf(**self._arguments)*dL_dK
                gradient = np.asarray([A[np.where(self._output_ind==i)].T.sum()
                                       for i in np.arange(self.output_dim)])
                if X2 is None:
                    gradient *= 2
                else:
                    A = gf(**self._reverse_arguments)*dL_dK.T
                    gradient += np.asarray([A[np.where(self._output_ind2==i)].T.sum()
                                 for i in np.arange(self.output_dim)])
                setattr(parameter, 'gradient', gradient)


    def update_gradients_diag(self, dL_dKdiag, X):
        self._K_computations(X)
        for theta in self._sym_theta:
            parameter = getattr(self, theta.name)
            gf = self._Kdiag_derivatives_code[theta.name]
            setattr(parameter, 'gradient', (gf(**self._diag_arguments)*dL_dKdiag).sum())
        if self.output_dim>1:
            for theta in self._sym_theta_i:
                parameter = getattr(self, theta.name[:-2])
                gf = self._Kdiag_derivatives_code[theta.name]
                a = gf(**self._diag_arguments)*dL_dKdiag
                setattr(parameter, 'gradient',
                        np.asarray([a[np.where(self._output_ind==i)].sum()
                         for i in np.arange(self.output_dim)]))

    def _K_computations(self, X, X2=None):
        """Set up argument lists for the derivatives."""
        # Could check if this needs doing or not, there could
        # definitely be some computational savings by checking for
        # parameter updates here.
        self._arguments = {}
        self._diag_arguments = {}
        for i, x in enumerate(self._sym_x):
            self._arguments[x.name] =  X[:, i][:, None]
            self._diag_arguments[x.name] =  X[:, i][:, None]
        if self.output_dim > 1:
            self._output_ind = np.asarray(X[:, -1], dtype='int')
            for i, theta in enumerate(self._sym_theta_i):
                self._arguments[theta.name] = np.asarray(getattr(self, theta.name[:-2])[self._output_ind])[:, None]
                self._diag_arguments[theta.name] = self._arguments[theta.name]
        for theta in self._sym_theta:
            self._arguments[theta.name] = np.asarray(getattr(self, theta.name))
            self._diag_arguments[theta.name] = self._arguments[theta.name]

        if X2 is not None:
            for i, z in enumerate(self._sym_z):
                self._arguments[z.name] =  X2[:, i][None, :]
            if self.output_dim > 1:
                self._output_ind2 = np.asarray(X2[:, -1], dtype='int')
                for i, theta in enumerate(self._sym_theta_j):
                    self._arguments[theta.name] = np.asarray(getattr(self, theta.name[:-2])[self._output_ind2])[None, :]
        else:
            for z in self._sym_z:
                self._arguments[z.name] =  self._arguments['x_'+z.name[2:]].T
            if self.output_dim > 1:
                self._output_ind2 = self._output_ind
                for theta in self._sym_theta_j:
                    self._arguments[theta.name] = self._arguments[theta.name[:-2] + '_i'].T
        if X2 is not None:
            # These arguments are needed in gradients when X2 is not equal to X.
            self._reverse_arguments = self._arguments
            for x, z in zip(self._sym_x, self._sym_z):
                self._reverse_arguments[x.name] = self._arguments[z.name].T
                self._reverse_arguments[z.name] = self._arguments[x.name].T
            if self.output_dim > 1:
                for theta_i, theta_j in zip(self._sym_theta_i, self._sym_theta_j):
                    self._reverse_arguments[theta_i.name] = self._arguments[theta_j.name].T
                    self._reverse_arguments[theta_j.name] = self._arguments[theta_i.name].T

if False:
    class Symcombine(CombinationKernel):
        """
        Combine list of given sympy covariances together with the provided operations.
        """
        def __init__(self, subkerns, operations, name='sympy_combine'):
            super(Symcombine, self).__init__(subkerns, name)
            for subkern, operation in zip(subkerns, operations):
                self._sym_k += self._k_double_operate(subkern._sym_k, operation)

        #def _double_operate(self, k, operation):


        @Cache_this(limit=2, force_kwargs=['which_parts'])
        def K(self, X, X2=None, which_parts=None):
            """
            Combine covariances with a linear operator.
            """
            assert X.shape[1] == self.input_dim
            if which_parts is None:
                which_parts = self.parts
            elif not isinstance(which_parts, (list, tuple)):
                # if only one part is given
                which_parts = [which_parts]
            return reduce(np.add, (p.K(X, X2) for p in which_parts))

        @Cache_this(limit=2, force_kwargs=['which_parts'])
        def Kdiag(self, X, which_parts=None):
            assert X.shape[1] == self.input_dim
            if which_parts is None:
                which_parts = self.parts
            elif not isinstance(which_parts, (list, tuple)):
                # if only one part is given
                which_parts = [which_parts]
            return reduce(np.add, (p.Kdiag(X) for p in which_parts))

        def update_gradients_full(self, dL_dK, X, X2=None):
            [p.update_gradients_full(dL_dK, X, X2) for p in self.parts]

        def update_gradients_diag(self, dL_dK, X):
            [p.update_gradients_diag(dL_dK, X) for p in self.parts]

        def gradients_X(self, dL_dK, X, X2=None):
            """Compute the gradient of the objective function with respect to X.

            :param dL_dK: An array of gradients of the objective function with respect to the covariance function.
            :type dL_dK: np.ndarray (num_samples x num_inducing)
            :param X: Observed data inputs
            :type X: np.ndarray (num_samples x input_dim)
            :param X2: Observed data inputs (optional, defaults to X)
            :type X2: np.ndarray (num_inducing x input_dim)"""

            target = np.zeros(X.shape)
            [target.__iadd__(p.gradients_X(dL_dK, X, X2)) for p in self.parts]
            return target

        def gradients_X_diag(self, dL_dKdiag, X):
            target = np.zeros(X.shape)
            [target.__iadd__(p.gradients_X_diag(dL_dKdiag, X)) for p in self.parts]
            return target

        def psi0(self, Z, variational_posterior):
            return reduce(np.add, (p.psi0(Z, variational_posterior) for p in self.parts))

        def psi1(self, Z, variational_posterior):
            return reduce(np.add, (p.psi1(Z, variational_posterior) for p in self.parts))

        def psi2(self, Z, variational_posterior):
            psi2 = reduce(np.add, (p.psi2(Z, variational_posterior) for p in self.parts))
            #return psi2
            # compute the "cross" terms
            from static import White, Bias
            from rbf import RBF
            #from rbf_inv import RBFInv
            from linear import Linear
            #ffrom fixed import Fixed

            for p1, p2 in itertools.combinations(self.parts, 2):
                # i1, i2 = p1.active_dims, p2.active_dims
                # white doesn;t combine with anything
                if isinstance(p1, White) or isinstance(p2, White):
                    pass
                # rbf X bias
                #elif isinstance(p1, (Bias, Fixed)) and isinstance(p2, (RBF, RBFInv)):
                elif isinstance(p1,  Bias) and isinstance(p2, (RBF, Linear)):
                    tmp = p2.psi1(Z, variational_posterior)
                    psi2 += p1.variance * (tmp[:, :, None] + tmp[:, None, :])
                #elif isinstance(p2, (Bias, Fixed)) and isinstance(p1, (RBF, RBFInv)):
                elif isinstance(p2, Bias) and isinstance(p1, (RBF, Linear)):
                    tmp = p1.psi1(Z, variational_posterior)
                    psi2 += p2.variance * (tmp[:, :, None] + tmp[:, None, :])
                elif isinstance(p2, (RBF, Linear)) and isinstance(p1, (RBF, Linear)):
                    assert np.intersect1d(p1.active_dims, p2.active_dims).size == 0, "only non overlapping kernel dimensions allowed so far"
                    tmp1 = p1.psi1(Z, variational_posterior)
                    tmp2 = p2.psi1(Z, variational_posterior)
                    psi2 += (tmp1[:, :, None] * tmp2[:, None, :]) + (tmp2[:, :, None] * tmp1[:, None, :])
                else:
                    raise NotImplementedError, "psi2 cannot be computed for this kernel"
            return psi2

        def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
            from static import White, Bias
            for p1 in self.parts:
                #compute the effective dL_dpsi1. Extra terms appear becaue of the cross terms in psi2!
                eff_dL_dpsi1 = dL_dpsi1.copy()
                for p2 in self.parts:
                    if p2 is p1:
                        continue
                    if isinstance(p2, White):
                        continue
                    elif isinstance(p2, Bias):
                        eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.variance * 2.
                    else:# np.setdiff1d(p1.active_dims, ar2, assume_unique): # TODO: Careful, not correct for overlapping active_dims
                        eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.psi1(Z, variational_posterior) * 2.
                p1.update_gradients_expectations(dL_dpsi0, eff_dL_dpsi1, dL_dpsi2, Z, variational_posterior)

        def gradients_Z_expectations(self, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
            from static import White, Bias
            target = np.zeros(Z.shape)
            for p1 in self.parts:
                #compute the effective dL_dpsi1. extra terms appear becaue of the cross terms in psi2!
                eff_dL_dpsi1 = dL_dpsi1.copy()
                for p2 in self.parts:
                    if p2 is p1:
                        continue
                    if isinstance(p2, White):
                        continue
                    elif isinstance(p2, Bias):
                        eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.variance * 2.
                    else:
                        eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.psi1(Z, variational_posterior) * 2.
                target += p1.gradients_Z_expectations(eff_dL_dpsi1, dL_dpsi2, Z, variational_posterior)
            return target

        def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
            from static import White, Bias
            target_mu = np.zeros(variational_posterior.shape)
            target_S = np.zeros(variational_posterior.shape)
            for p1 in self._parameters_:
                #compute the effective dL_dpsi1. extra terms appear becaue of the cross terms in psi2!
                eff_dL_dpsi1 = dL_dpsi1.copy()
                for p2 in self._parameters_:
                    if p2 is p1:
                        continue
                    if isinstance(p2, White):
                        continue
                    elif isinstance(p2, Bias):
                        eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.variance * 2.
                    else:
                        eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.psi1(Z, variational_posterior) * 2.
                a, b = p1.gradients_qX_expectations(dL_dpsi0, eff_dL_dpsi1, dL_dpsi2, Z, variational_posterior)
                target_mu += a
                target_S += b
            return target_mu, target_S

        def _getstate(self):
            """
            Get the current state of the class,
            here just all the indices, rest can get recomputed
            """
            return super(Add, self)._getstate()

        def _setstate(self, state):
            super(Add, self)._setstate(state)

        def add(self, other, name='sum'):
            if isinstance(other, Add):
                other_params = other._parameters_.copy()
                for p in other_params:
                    other.remove_parameter(p)
                self.add_parameters(*other_params)
            else: self.add_parameter(other)
            return self
