# Check Matthew Rocklin's blog post.
try: 
    import sympy as sp
    sympy_available=True
    from sympy.utilities.lambdify import lambdify
except ImportError:
    sympy_available=False
    exit()

import numpy as np
from kern import Kern
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp

class Sympykern(Kern):
    """
    A kernel object, where all the hard work in done by sympy.

    :param k: the covariance function
    :type k: a positive definite sympy function of x_0, z_0, x_1, z_1, x_2, z_2...

    To construct a new sympy kernel, you'll need to define:
     - a kernel function using a sympy object. Ensure that the kernel is of the form k(x,z).
     - that's it! we'll extract the variables from the function k.

    Note:
     - to handle multiple inputs, call them x_1, z_1, etc
     - to handle multpile correlated outputs, you'll need to add parameters with an index, such as lengthscale_i and lengthscale_j.
    """
    def __init__(self, input_dim, k=None, output_dim=1, name=None, param=None):

        if name is None:
            name='sympykern'
        if k is None:
            raise ValueError, "You must provide an argument for the covariance function."
        super(Sympykern, self).__init__(input_dim, name)

        self._sp_k = k
        
        # pull the variable names out of the symbolic covariance function.
        sp_vars = [e for e in k.atoms() if e.is_Symbol]
        self._sp_x= sorted([e for e in sp_vars if e.name[0:2]=='x_'],key=lambda x:int(x.name[2:]))
        self._sp_z= sorted([e for e in sp_vars if e.name[0:2]=='z_'],key=lambda z:int(z.name[2:]))

        # Check that variable names make sense.
        assert all([x.name=='x_%i'%i for i,x in enumerate(self._sp_x)])
        assert all([z.name=='z_%i'%i for i,z in enumerate(self._sp_z)])
        assert len(self._sp_x)==len(self._sp_z)
        x_dim=len(self._sp_x)

        self._sp_kdiag = k
        for x, z in zip(self._sp_x, self._sp_z):
            self._sp_kdiag = self._sp_kdiag.subs(z, x)
            
        # If it is a multi-output covariance, add an input for indexing the outputs.
        self._real_input_dim = x_dim
        # Check input dim is number of xs + 1 if output_dim is >1
        assert self.input_dim == x_dim + int(output_dim > 1)
        self.output_dim = output_dim

        # extract parameter names from the covariance
        thetas = sorted([e for e in sp_vars if not (e.name[0:2]=='x_' or e.name[0:2]=='z_')],key=lambda e:e.name)


        # Look for parameters with index (subscripts), they are associated with different outputs.
        if self.output_dim>1:
            self._sp_theta_i = sorted([e for e in thetas if (e.name[-2:]=='_i')], key=lambda e:e.name)
            self._sp_theta_j = sorted([e for e in thetas if (e.name[-2:]=='_j')], key=lambda e:e.name)

            # Make sure parameter appears with both indices!
            assert len(self._sp_theta_i)==len(self._sp_theta_j)
            assert all([theta_i.name[:-2]==theta_j.name[:-2] for theta_i, theta_j in zip(self._sp_theta_i, self._sp_theta_j)])

            # Extract names of shared parameters (those without a subscript)
            self._sp_theta = [theta for theta in thetas if theta not in self._sp_theta_i and theta not in self._sp_theta_j]
            
            self.num_split_params = len(self._sp_theta_i)
            self._split_theta_names = ["%s"%theta.name[:-2] for theta in self._sp_theta_i]
            # Add split parameters to the model.
            for theta in self._split_theta_names:
                # TODO: what if user has passed a parameter vector, how should that be stored and interpreted?
                setattr(self, theta, Param(theta, np.ones(self.output_dim), None))
                self.add_parameter(getattr(self, theta))

            
            self.num_shared_params = len(self._sp_theta)
            for theta_i, theta_j in zip(self._sp_theta_i, self._sp_theta_j):
                self._sp_kdiag = self._sp_kdiag.subs(theta_j, theta_i)
            
        else:
            self.num_split_params = 0
            self._split_theta_names = []
            self._sp_theta = thetas
            self.num_shared_params = len(self._sp_theta)

        # Add parameters to the model.
        for theta in self._sp_theta:
            val = 1.0
            # TODO: what if user has passed a parameter vector, how should that be stored and interpreted? This is the old way before params class.
            if param is not None:
                if param.has_key(theta):
                    val = param[theta]
            setattr(self, theta.name, Param(theta.name, val, None))
            self.add_parameters(getattr(self, theta.name))

        # Differentiate with respect to parameters.
        derivative_arguments = self._sp_x + self._sp_theta
        if self.output_dim > 1:
            derivative_arguments += self._sp_theta_i
        
        self.derivatives = {theta.name : sp.diff(self._sp_k,theta).simplify() for theta in derivative_arguments}
        self.diag_derivatives = {theta.name : sp.diff(self._sp_kdiag,theta).simplify() for theta in derivative_arguments}
        
        # This gives the parameters for the arg list.
        self.arg_list = self._sp_x + self._sp_z + self._sp_theta
        self.diag_arg_list = self._sp_x + self._sp_theta
        if self.output_dim > 1:
            self.arg_list += self._sp_theta_i + self._sp_theta_j
            self.diag_arg_list += self._sp_theta_i
        # psi_stats aren't yet implemented.
        if False:
            self.compute_psi_stats()

        # generate the code for the covariance functions
        self._gen_code()

    def __add__(self,other):
        return spkern(self._sp_k+other._sp_k)

    def _gen_code(self):
        #fn_theano = theano_function([self.arg_lists], [self._sp_k + self.derivatives], dims={x: 1}, dtypes={x_0: 'float64', z_0: 'float64'})
        self._K_function = lambdify(self.arg_list, self._sp_k, 'numpy')
        for key in self.derivatives.keys():
            setattr(self, '_K_diff_' + key, lambdify(self.arg_list, self.derivatives[key], 'numpy'))

        self._Kdiag_function = lambdify(self.diag_arg_list, self._sp_kdiag, 'numpy')
        for key in self.derivatives.keys():
            setattr(self, '_Kdiag_diff_' + key, lambdify(self.diag_arg_list, self.diag_derivatives[key], 'numpy'))

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
        for i, x in enumerate(self._sp_x):
            gf = getattr(self, '_K_diff_' + x.name)
            gradients_X[:, i] = (gf(**self._arguments)*dL_dK).sum(1)
        if X2 is None:
            gradients_X *= 2
        return gradients_X

    def gradients_X_diag(self, dL_dK, X):
        self._K_computations(X)
        dX = np.zeros_like(X)
        for i, x in enumerate(self._sp_x):
            gf = getattr(self, '_Kdiag_diff_' + x.name)
            dX[:, i] = gf(**self._diag_arguments)*dL_dK
        return dX
    
    def update_gradients_full(self, dL_dK, X, X2=None):
        # Need to extract parameters to local variables first
        self._K_computations(X, X2)
        for theta in self._sp_theta:
            parameter = getattr(self, theta.name)
            gf = getattr(self, '_K_diff_' + theta.name)
            gradient = (gf(**self._arguments)*dL_dK).sum()
            if X2 is not None:
                gradient += (gf(**self._reverse_arguments)*dL_dK).sum()
            setattr(parameter, 'gradient', gradient)
        if self.output_dim>1:
            for theta in self._sp_theta_i:
                parameter = getattr(self, theta.name[:-2])
                gf = getattr(self, '_K_diff_' + theta.name)
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
        for theta in self._sp_theta:
            parameter = getattr(self, theta.name)
            gf = getattr(self, '_Kdiag_diff_' + theta.name)
            setattr(parameter, 'gradient', (gf(**self._diag_arguments)*dL_dKdiag).sum())
        if self.output_dim>1:
            for theta in self._sp_theta_i:
                parameter = getattr(self, theta.name[:-2])
                gf = getattr(self, '_Kdiag_diff_' + theta.name)
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
        for i, x in enumerate(self._sp_x):
            self._arguments[x.name] =  X[:, i][:, None]
            self._diag_arguments[x.name] =  X[:, i][:, None]
        if self.output_dim > 1:
            self._output_ind = np.asarray(X[:, -1], dtype='int')
            for i, theta in enumerate(self._sp_theta_i):
                self._arguments[theta.name] = np.asarray(getattr(self, theta.name[:-2])[self._output_ind])[:, None]
                self._diag_arguments[theta.name] = self._arguments[theta.name]
        for theta in self._sp_theta:
            self._arguments[theta.name] = np.asarray(getattr(self, theta.name))
            self._diag_arguments[theta.name] = self._arguments[theta.name]

        if X2 is not None:
            for i, z in enumerate(self._sp_z):
                self._arguments[z.name] =  X2[:, i][None, :]
            if self.output_dim > 1:
                self._output_ind2 = np.asarray(X2[:, -1], dtype='int')
                for i, theta in enumerate(self._sp_theta_j):
                    self._arguments[theta.name] = np.asarray(getattr(self, theta.name[:-2])[self._output_ind2])[None, :]
        else:
            for z in self._sp_z:
                self._arguments[z.name] =  self._arguments['x_'+z.name[2:]].T
            if self.output_dim > 1:
                self._output_ind2 = self._output_ind
                for theta in self._sp_theta_j:
                    self._arguments[theta.name] = self._arguments[theta.name[:-2] + '_i'].T
        if X2 is not None:
            # These arguments are needed in gradients when X2 is not equal to X.
            self._reverse_arguments = self._arguments
            for x, z in zip(self._sp_x, self._sp_z):
                self._reverse_arguments[x.name] = self._arguments[z.name].T
                self._reverse_arguments[z.name] = self._arguments[x.name].T
            if self.output_dim > 1:
                for theta_i, theta_j in zip(self._sp_theta_i, self._sp_theta_j):
                    self._reverse_arguments[theta_i.name] = self._arguments[theta_j.name].T
                    self._reverse_arguments[theta_j.name] = self._arguments[theta_i.name].T

