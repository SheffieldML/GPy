********************
Creating new kernels
********************

We will see in this tutorial how to create new kernels in GPy. We will also give details on how to implement each function of the kernel and illustrate with a running example: the rational quadratic kernel. 

Structure of a kernel in GPy
============================

In GPy a kernel object is made of a list of kernpart objects, which correspond to symetric positive definite functions. More precisely, the kernel should be understood as the sum of the kernparts. In order to implement a new covariance, you must implement the new covariance as a :py:class:`GPy.kern.src.kern.Kern` object. This is detailed below.

Implementing a Kern object
==============================

We advise the reader to start with copy-pasting an existing kernel and
to modify the new file. We will now give a description of the various
functions that can be found in a Kern object, some of which are
mandatory for the new kernel to work.

Header
~~~~~~

The header is similar to all kernels: ::

    from .kern import Kern
    import numpy as np
    from GPy.core.parameterization.param import Param

    class RationalQuadratic(Kern):

:py:func:`GPy.kern.src.kern.Kern.__init__` ``(self, input_dim, param1, param2, *args)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
The implementation of this function in mandatory.

For all Kerns the first parameter ``input_dim`` corresponds to the
dimension of the input space, and the following parameters stand for
the parameterization of the kernel.

You have to call ``super(<class_name>, self).__init__(input_dim, active_dims, 
name)`` to make sure the input dimension (and possible dimension restrictions using active_dims) and name of the kernel are
stored in the right place. These attributes are available as
``self.input_dim`` and ``self.name`` at runtime.  Parameterization is
done by adding :py:class:`~GPy.core.parameterization.param.Param`
objects to ``self`` and use them as normal numpy ``array-like`` s in
your code. The parameters have to be added by calling
:py:func:`~GPy.core.parameterization.parameterized.Parameterized.link_parameters`
``(*parameters)`` with the
:py:class:`~GPy.core.parameterization.param.Param` objects as
arguments::

    def __init__(self,input_dim,variance=1.,lengthscale=1.,power=1.,active_dims=None):
        super(RationalQuadratic, self).__init__(input_dim, active_dims, 'rat_quad')
	assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.variance = Param('variance', variance)
        self.lengthscale = Param('lengtscale', lengthscale)
        self.power = Param('power', power)
	self.link_parameters(self.variance, self.lengthscale, self.power)

From now on you can use the parameters ``self.variance,
self.lengthscale, self.power`` as normal numpy ``array-like`` s in your
code. Updates from the optimization routine will be done
automatically.

:py:func:`~GPy.core.parameterization.parameter_core.Parameterizable.parameters_changed` ``(self)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation of this function is optional.

This functions deals as a callback for each optimization iteration. If
one optimization step was successfull and the parameters (added by
:py:func:`~GPy.core.parameterization.parameterized.Parameterized.link_parameters`
``(*parameters)``) this callback function will be called to be able to
update any precomputations for the kernel. Do not implement the
gradient updates here, as those are being done by the model enclosing
the kernel::

    def parameters_changed(self):
        # nothing todo here
	pass


:py:func:`~GPy.kern.src.kern.Kern.K` ``(self,X,X2)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation of this function in mandatory.

This function is used to compute the covariance matrix associated with
the inputs X, X2 (np.arrays with arbitrary number of line (say
:math:`n_1`, :math:`n_2`) and ``self.input_dim`` columns). ::

    def K(self,X,X2):
        if X2 is None: X2 = X
        dist2 = np.square((X-X2.T)/self.lengthscale)
        return self.variance*(1 + dist2/2.)**(-self.power)

:py:func:`~GPy.kern.src.kern.Kern.Kdiag` ``(self,X)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation of this function is mandatory.

This function is similar to ``K`` but it computes only the values of
the kernel on the diagonal. Thus, ``target`` is a 1-dimensional
np.array of length :math:`n \times 1`. ::

    def Kdiag(self,X):
        return self.variance*np.ones(X.shape[0])

:py:func:`~GPy.kern.src.kern.Kern.update_gradients_full` ``(self, dL_dK, X, X2=None)``
~~~~~~~~~~~~~~~~~~~

This function is required for the optimization of the parameters.

Computes the gradients and sets them on the parameters of this model.
For example, if the kernel is parameterized by
:math:`\sigma^2, \theta`, then

.. math::

   \frac{\partial L}{\partial\sigma^2}
    = \frac{\partial L}{\partial K} \frac{\partial K}{\partial\sigma^2}

is added to the gradient of :math:`\sigma^2`: ``self.variance.gradient = <gradient>``
and

.. math::

   \frac{\partial L}{\partial\theta}
    = \frac{\partial L}{\partial K} \frac{\partial K}{\partial\theta}

to :math:`\theta`. ::
	  
    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X
        dist2 = np.square((X-X2.T)/self.lengthscale)

        dvar = (1 + dist2/2.)**(-self.power)
        dl = self.power * self.variance * dist2 * self.lengthscale**(-3) * (1 + dist2/2./self.power)**(-self.power-1)
        dp = - self.variance * np.log(1 + dist2/2.) * (1 + dist2/2.)**(-self.power)

        self.variance.gradient = np.sum(dvar*dL_dK)
        self.lengthscale.gradient = np.sum(dl*dL_dK)
        self.power.gradient = np.sum(dp*dL_dK)


:py:func:`~GPy.kern.src.kern.Kern.update_gradients_diag` ``(self,dL_dKdiag,X,target)``
~~~~~~~~~~~~~~~~~~~
    
This function is required for BGPLVM, sparse models and uncertain inputs.

As previously, target is an ``self.num_params`` array and

.. math::

   \frac{\partial L}{\partial Kdiag}
    \frac{\partial Kdiag}{\partial param}

is set to each ``param``. ::

    def update_gradients_diag(self, dL_dKdiag, X):
        self.variance.gradient = np.sum(dL_dKdiag)
        # here self.lengthscale and self.power have no influence on Kdiag so target[1:] are unchanged

:py:func:`~GPy.kern.src.kern.Kern.gradients_X` ``(self,dL_dK, X, X2)``
~~~~~~~~~~~~~~~~~~~

This function is required for GPLVM, BGPLVM, sparse models and uncertain inputs.

Computes the derivative of the likelihood with respect to the inputs
``X`` (a :math:`n \times q` np.array). The result is returned by the
function which is a :math:`n \times q` np.array. ::

    def gradients_X(self,dL_dK,X,X2):
        """derivative of the covariance matrix with respect to X."""
        if X2 is None: X2 = X
        dist2 = np.square((X-X2.T)/self.lengthscale)

        dX = -self.variance*self.power * (X-X2.T)/self.lengthscale**2 *  (1 + dist2/2./self.lengthscale)**(-self.power-1)
        return np.sum(dL_dK*dX,1)[:,None]

:py:func:`~GPy.kern.src.kern.Kern.gradients_X_diag` ``(self,dL_dKdiag,X)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
This function is required for BGPLVM, sparse models and uncertain
inputs. As for ``dKdiag_dtheta``,

.. math::

   \frac{\partial L}{\partial Kdiag} \frac{\partial Kdiag}{\partial X}

is added to each element of target. ::

    def gradients_X_diag(self,dL_dKdiag,X):
        # no diagonal gradients
        pass

**Second order derivatives**
~~~~~~~~~~~~~~~~~~~~~~~~

These functions are required for the magnification factor and are the same as the first order gradients for X, but
as the second order derivatives:

.. math:: \frac{\partial^2 K}{\partial X\partial X2}

- :py:func:`GPy.kern.src.kern.gradients_XX` ``(self,dL_dK, X, X2)``
- :py:func:`GPy.kern.src.kern.gradients_XX_diag` ``(self,dL_dKdiag, X)``
	
**Psi statistics**
~~~~~~~~~~~~~

The psi statistics and their derivatives are required for BGPLVM and
GPS with uncertain inputs only, the expressions are as follows

- `psi0(self, Z, variational_posterior)`
   .. math::

     \psi_0 = \sum_{i=0}^{n}E_{q(X)}[k(X_i, X_i)]

- `psi1(self, Z, variational_posterior)`::
   .. math::

      \psi_1^{n,m} = E_{q(X)}[k(X_n, Z_m)]
	
- `psi2(self, Z, variational_posterior)`
   .. math::

      \psi_2^{m,m'} = \sum_{i=0}^{n}E_{q(X)}[ k(Z_m, X_i) k(X_i, Z_{m'})]
	
- `psi2n(self, Z, variational_posterior)`
   .. math::

      \psi_2^{n,m,m'} = E_{q(X)}[ k(Z_m, X_n) k(X_n, Z_{m'})]
