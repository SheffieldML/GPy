# Copyright (c) 2012, 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from .. import likelihoods
from ..inference import optimization
from ..util.misc import opt_wrapper
from parameterization import Parameterized
import multiprocessing as mp
import numpy as np
from numpy.linalg.linalg import LinAlgError
import itertools
# import numdifftools as ndt

class Model(Parameterized):
    _fail_count = 0  # Count of failed optimization steps (see objective)
    _allowed_failures = 10  # number of allowed failures

    def __init__(self, name):
        super(Model, self).__init__(name)  # Parameterized.__init__(self)
        self.optimization_runs = []
        self.sampling_runs = []
        self.preferred_optimizer = 'scg'

    def log_likelihood(self):
        raise NotImplementedError, "this needs to be implemented to use the model class"

    def _log_likelihood_gradients(self):
        return self.gradient

    def _getstate(self):
        """
        Get the current state of the class.
        Inherited from Parameterized, so add those parameters to the state

        :return: list of states from the model.

        """
        return Parameterized._getstate(self) + \
            [self.priors, self.optimization_runs,
             self.sampling_runs, self.preferred_optimizer]

    def _setstate(self, state):
        """
        set state from previous call to _getstate
        call Parameterized with the rest of the state

        :param state: the state of the model.
        :type state: list as returned from _getstate.

        """
        self.preferred_optimizer = state.pop()
        self.sampling_runs = state.pop()
        self.optimization_runs = state.pop()
        self.priors = state.pop()
        Parameterized._setstate(self, state)

    def optimize_restarts(self, num_restarts=10, robust=False, verbose=True, parallel=False, num_processes=None, **kwargs):
        """
        Perform random restarts of the model, and set the model to the best
        seen solution.

        If the robust flag is set, exceptions raised during optimizations will
        be handled silently.  If _all_ runs fail, the model is reset to the
        existing parameter values.

        **Notes**

        :param num_restarts: number of restarts to use (default 10)
        :type num_restarts: int
        :param robust: whether to handle exceptions silently or not (default False)
        :type robust: bool
        :param parallel: whether to run each restart as a separate process. It relies on the multiprocessing module.
        :type parallel: bool
        :param num_processes: number of workers in the multiprocessing pool
        :type numprocesses: int

        \*\*kwargs are passed to the optimizer. They can be:

        :param max_f_eval: maximum number of function evaluations
        :type max_f_eval: int
        :param max_iters: maximum number of iterations
        :type max_iters: int
        :param messages: whether to display during optimisation
        :type messages: bool

        .. note:: If num_processes is None, the number of workes in the
        multiprocessing pool is automatically set to the number of processors
        on the current machine.

        """
        initial_parameters = self._get_params_transformed()

        if parallel:
            try:
                jobs = []
                pool = mp.Pool(processes=num_processes)
                for i in range(num_restarts):
                    self.randomize()
                    job = pool.apply_async(opt_wrapper, args=(self,), kwds=kwargs)
                    jobs.append(job)

                pool.close()  # signal that no more data coming in
                pool.join()  # wait for all the tasks to complete
            except KeyboardInterrupt:
                print "Ctrl+c received, terminating and joining pool."
                pool.terminate()
                pool.join()

        for i in range(num_restarts):
            try:
                if not parallel:
                    self.randomize()
                    self.optimize(**kwargs)
                else:
                    self.optimization_runs.append(jobs[i].get())

                if verbose:
                    print("Optimization restart {0}/{1}, f = {2}".format(i + 1, num_restarts, self.optimization_runs[-1].f_opt))
            except Exception as e:
                if robust:
                    print("Warning - optimization restart {0}/{1} failed".format(i + 1, num_restarts))
                else:
                    raise e

        if len(self.optimization_runs):
            i = np.argmin([o.f_opt for o in self.optimization_runs])
            self._set_params_transformed(self.optimization_runs[i].x_opt)
        else:
            self._set_params_transformed(initial_parameters)

    def ensure_default_constraints(self, warning=True):
        """
        Ensure that any variables which should clearly be positive
        have been constrained somehow. The method performs a regular
        expression search on parameter names looking for the terms
        'variance', 'lengthscale', 'precision' and 'kappa'. If any of
        these terms are present in the name the parameter is
        constrained positive.

        DEPRECATED.
        """
        raise DeprecationWarning, 'parameters now have default constraints'

    def input_sensitivity(self):
        """
        Returns the sensitivity for each dimension of this kernel.
        """
        return self.kern.input_sensitivity()

    def objective_function(self, x):
        """
        The objective function passed to the optimizer. It combines
        the likelihood and the priors.

        Failures are handled robustly. The algorithm will try several times to
        return the objective, and will raise the original exception if
        the objective cannot be computed.

        :param x: the parameters of the model.
        :parameter type: np.array
        """
        try:
            self._set_params_transformed(x)
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError) as e:
            if self._fail_count >= self._allowed_failures:
                raise e
            self._fail_count += 1
            return np.inf
        return -float(self.log_likelihood()) - self.log_prior()

    def objective_function_gradients(self, x):
        """
        Gets the gradients from the likelihood and the priors.

        Failures are handled robustly. The algorithm will try several times to
        return the gradients, and will raise the original exception if
        the objective cannot be computed.

        :param x: the parameters of the model.
        :type x: np.array
        """
        try:
            self._set_params_transformed(x)
            obj_grads = -self._transform_gradients(self._log_likelihood_gradients() + self._log_prior_gradients())
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError) as e:
            if self._fail_count >= self._allowed_failures:
                raise e
            self._fail_count += 1
            obj_grads = np.clip(-self._transform_gradients(self._log_likelihood_gradients() + self._log_prior_gradients()), -1e100, 1e100)
        return obj_grads

    def objective_and_gradients(self, x):
        """
        Compute the objective function of the model and the gradient of the model at the point given by x.

        :param x: the point at which gradients are to be computed.
        :type x: np.array
        """

        try:
            self._set_params_transformed(x)
            obj_f = -float(self.log_likelihood()) - self.log_prior()
            obj_grads = -self._transform_gradients(self._log_likelihood_gradients() + self._log_prior_gradients())
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError) as e:
            if self._fail_count >= self._allowed_failures:
                raise e
            self._fail_count += 1
            obj_f = np.inf
            obj_grads = np.clip(-self._transform_gradients(self._log_likelihood_gradients() + self._log_prior_gradients()), -1e100, 1e100)
        return obj_f, obj_grads

    def optimize(self, optimizer=None, start=None, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.
        kwargs are passed to the optimizer. They can be:

        :param max_f_eval: maximum number of function evaluations
        :type max_f_eval: int
        :messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred optimizer)
        :type optimizer: string

        TODO: valid args
        """
        if self.is_fixed:
            raise RuntimeError, "Cannot optimize, when everything is fixed"
        if self.size == 0:
            raise RuntimeError, "Model without parameters cannot be minimized"

        if optimizer is None:
            optimizer = self.preferred_optimizer

        if start == None:
            start = self._get_params_transformed()

        optimizer = optimization.get_optimizer(optimizer)
        opt = optimizer(start, model=self, **kwargs)

        opt.run(f_fp=self.objective_and_gradients, f=self.objective_function, fp=self.objective_function_gradients)

        self.optimization_runs.append(opt)

        self._set_params_transformed(opt.x_opt)

    def optimize_SGD(self, momentum=0.1, learning_rate=0.01, iterations=20, **kwargs):
        # assert self.Y.shape[1] > 1, "SGD only works with D > 1"
        sgd = SGD.StochasticGD(self, iterations, learning_rate, momentum, **kwargs)  # @UndefinedVariable
        sgd.run()
        self.optimization_runs.append(sgd)

    def _checkgrad(self, target_param=None, verbose=False, step=1e-6, tolerance=1e-3):
        """
        Check the gradient of the ,odel by comparing to a numerical
        estimate.  If the verbose flag is passed, invividual
        components are tested (and printed)

        :param verbose: If True, print a "full" checking of each parameter
        :type verbose: bool
        :param step: The size of the step around which to linearise the objective
        :type step: float (default 1e-6)
        :param tolerance: the tolerance allowed (see note)
        :type tolerance: float (default 1e-3)

        Note:-
           The gradient is considered correct if the ratio of the analytical
           and numerical gradients is within <tolerance> of unity.
        """
        x = self._get_params_transformed().copy()

        if not verbose:
            # make sure only to test the selected parameters
            if target_param is None:
                transformed_index = range(len(x))
            else:
                transformed_index = self._raveled_index_for(target_param)
                if self._has_fixes():
                    indices = np.r_[:self.size]
                    which = (transformed_index[:,None]==indices[self._fixes_][None,:]).nonzero()
                    transformed_index = (indices-(~self._fixes_).cumsum())[transformed_index[which[0]]]

                if transformed_index.size == 0:
                    print "No free parameters to check"
                    return

            # just check the global ratio
            dx = np.zeros(x.shape)
            dx[transformed_index] = step * np.sign(np.random.uniform(-1, 1, transformed_index.size))

            # evaulate around the point x
            f1 = self.objective_function(x + dx)
            f2 = self.objective_function(x - dx)
            gradient = self.objective_function_gradients(x)

            dx = dx[transformed_index]
            gradient = gradient[transformed_index]

            denominator = (2 * np.dot(dx, gradient))
            global_ratio = (f1 - f2) / np.where(denominator==0., 1e-32, denominator)
            return np.abs(1. - global_ratio) < tolerance or np.abs(f1-f2).sum() + np.abs((2 * np.dot(dx, gradient))).sum() < tolerance
        else:
            # check the gradient of each parameter individually, and do some pretty printing
            try:
                names = self._get_param_names()
            except NotImplementedError:
                names = ['Variable %i' % i for i in range(len(x))]
            # Prepare for pretty-printing
            header = ['Name', 'Ratio', 'Difference', 'Analytical', 'Numerical']
            max_names = max([len(names[i]) for i in range(len(names))] + [len(header[0])])
            float_len = 10
            cols = [max_names]
            cols.extend([max(float_len, len(header[i])) for i in range(1, len(header))])
            cols = np.array(cols) + 5
            header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
            header_string = map(lambda x: '|'.join(x), [header_string])
            separator = '-' * len(header_string[0])
            print '\n'.join([header_string[0], separator])
            if target_param is None:
                param_index = range(len(x))
                transformed_index = param_index
            else:
                param_index = self._raveled_index_for(target_param)
                if self._has_fixes():
                    indices = np.r_[:self.size]
                    which = (param_index[:,None]==indices[self._fixes_][None,:]).nonzero()
                    param_index = param_index[which[0]]
                    transformed_index = (indices-(~self._fixes_).cumsum())[param_index]
                    #print param_index, transformed_index
                else:
                    transformed_index = param_index

                if param_index.size == 0:
                    print "No free parameters to check"
                    return

            gradient = self.objective_function_gradients(x).copy()
            np.where(gradient == 0, 1e-312, gradient)
            ret = True
            for nind, xind in itertools.izip(param_index, transformed_index):
                xx = x.copy()
                xx[xind] += step
                f1 = self.objective_function(xx)
                xx[xind] -= 2.*step
                f2 = self.objective_function(xx)
                numerical_gradient = (f1 - f2) / (2 * step)
                if np.all(gradient[xind]==0): ratio = (f1-f2) == gradient[xind]
                else: ratio = (f1 - f2) / (2 * step * gradient[xind])
                difference = np.abs((f1 - f2) / 2 / step - gradient[xind])

                if (np.abs(1. - ratio) < tolerance) or np.abs(difference) < tolerance:
                    formatted_name = "\033[92m {0} \033[0m".format(names[nind])
                    ret &= True
                else:
                    formatted_name = "\033[91m {0} \033[0m".format(names[nind])
                    ret &= False

                r = '%.6f' % float(ratio)
                d = '%.6f' % float(difference)
                g = '%.6f' % gradient[xind]
                ng = '%.6f' % float(numerical_gradient)
                grad_string = "{0:<{c0}}|{1:^{c1}}|{2:^{c2}}|{3:^{c3}}|{4:^{c4}}".format(formatted_name, r, d, g, ng, c0=cols[0] + 9, c1=cols[1], c2=cols[2], c3=cols[3], c4=cols[4])
                print grad_string

            self._set_params_transformed(x)
            return ret


