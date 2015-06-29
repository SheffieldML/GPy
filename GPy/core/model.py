# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from .. import likelihoods
from ..inference import optimization
from ..util.misc import opt_wrapper
from .parameterization import Parameterized
import multiprocessing as mp
import numpy as np
from numpy.linalg.linalg import LinAlgError
import itertools
import sys
from .verbose_optimization import VerboseOptimization
# import numdifftools as ndt
from functools import reduce

class Model(Parameterized):
    _fail_count = 0  # Count of failed optimization steps (see objective)
    _allowed_failures = 10  # number of allowed failures

    def __init__(self, name):
        super(Model, self).__init__(name)  # Parameterized.__init__(self)
        self.optimization_runs = []
        self.sampling_runs = []
        self.preferred_optimizer = 'bfgs'
        from .parameterization.ties_and_remappings import Tie
        self.tie = Tie()
        self.link_parameter(self.tie, -1)
        self.obj_grads = None
        self.add_observer(self.tie, self.tie._parameters_changed_notification, priority=-500)

    def log_likelihood(self):
        raise NotImplementedError("this needs to be implemented to use the model class")
    def _log_likelihood_gradients(self):
        return self.gradient.copy()

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
        initial_parameters = self.optimizer_array.copy()

        if parallel:
            try:
                jobs = []
                pool = mp.Pool(processes=num_processes)
                for i in range(num_restarts):
                    if i>0: self.randomize()
                    job = pool.apply_async(opt_wrapper, args=(self,), kwds=kwargs)
                    jobs.append(job)

                pool.close()  # signal that no more data coming in
                pool.join()  # wait for all the tasks to complete
            except KeyboardInterrupt:
                print("Ctrl+c received, terminating and joining pool.")
                pool.terminate()
                pool.join()

        for i in range(num_restarts):
            try:
                if not parallel:
                    if i>0: self.randomize()
                    self.optimize(**kwargs)
                else:
                    self.optimization_runs.append(jobs[i].get())

                if verbose:
                    print(("Optimization restart {0}/{1}, f = {2}".format(i + 1, num_restarts, self.optimization_runs[-1].f_opt)))
            except Exception as e:
                if robust:
                    print(("Warning - optimization restart {0}/{1} failed".format(i + 1, num_restarts)))
                else:
                    raise e

        if len(self.optimization_runs):
            i = np.argmin([o.f_opt for o in self.optimization_runs])
            self.optimizer_array = self.optimization_runs[i].x_opt
        else:
            self.optimizer_array = initial_parameters

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
        raise DeprecationWarning('parameters now have default constraints')

    def objective_function(self):
        """
        The objective function for the given algorithm.

        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the objective function here.

        For probabilistic models this is the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your objective to minimize here!
        """
        return -float(self.log_likelihood()) - self.log_prior()

    def objective_function_gradients(self):
        """
        The gradients for the objective function for the given algorithm.
        The gradients are w.r.t. the *negative* objective function, as
        this framework works with *negative* log-likelihoods as a default.

        You can find the gradient for the parameters in self.gradient at all times.
        This is the place, where gradients get stored for parameters.

        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the gradient here.

        For probabilistic models this is the gradient of the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your *negative* gradient here!
        """
        return -(self._log_likelihood_gradients() + self._log_prior_gradients())

    def _grads(self, x):
        """
        Gets the gradients from the likelihood and the priors.

        Failures are handled robustly. The algorithm will try several times to
        return the gradients, and will raise the original exception if
        the objective cannot be computed.

        :param x: the parameters of the model.
        :type x: np.array
        """
        try:
            # self._set_params_transformed(x)
            self.optimizer_array = x
            self.obj_grads = self._transform_gradients(self.objective_function_gradients())
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError):
            if self._fail_count >= self._allowed_failures:
                raise
            self._fail_count += 1
            self.obj_grads = np.clip(self._transform_gradients(self.objective_function_gradients()), -1e100, 1e100)
        return self.obj_grads

    def _objective(self, x):
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
            self.optimizer_array = x
            obj = self.objective_function()
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError):
            if self._fail_count >= self._allowed_failures:
                raise
            self._fail_count += 1
            return np.inf
        return obj

    def _objective_grads(self, x):
        try:
            self.optimizer_array = x
            obj_f, self.obj_grads = self.objective_function(), self._transform_gradients(self.objective_function_gradients())
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError):
            if self._fail_count >= self._allowed_failures:
                raise
            self._fail_count += 1
            obj_f = np.inf
            self.obj_grads = np.clip(self._transform_gradients(self.objective_function_gradients()), -1e10, 1e10)
        return obj_f, self.obj_grads

    def optimize(self, optimizer=None, start=None, messages=False, max_iters=1000, ipython_notebook=True, clear_after_finish=False, **kwargs):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.

        kwargs are passed to the optimizer. They can be:

        :param max_iters: maximum number of function evaluations
        :type max_iters: int
        :messages: True: Display messages during optimisation, "ipython_notebook":
        :type messages: bool"string
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
            start = self.optimizer_array

        if optimizer is None:
            optimizer = self.preferred_optimizer

        if isinstance(optimizer, optimization.Optimizer):
            opt = optimizer
            opt.model = self
        else:
            optimizer = optimization.get_optimizer(optimizer)
            opt = optimizer(start, model=self, max_iters=max_iters, **kwargs)

        with VerboseOptimization(self, opt, maxiters=max_iters, verbose=messages, ipython_notebook=ipython_notebook, clear_after_finish=clear_after_finish) as vo:
            opt.run(f_fp=self._objective_grads, f=self._objective, fp=self._grads)
            vo.finish(opt)

        self.optimization_runs.append(opt)

        self.optimizer_array = opt.x_opt

    def optimize_SGD(self, momentum=0.1, learning_rate=0.01, iterations=20, **kwargs):
        # assert self.Y.shape[1] > 1, "SGD only works with D > 1"
        sgd = SGD.StochasticGD(self, iterations, learning_rate, momentum, **kwargs)  # @UndefinedVariable
        sgd.run()
        self.optimization_runs.append(sgd)

    def _checkgrad(self, target_param=None, verbose=False, step=1e-6, tolerance=1e-3, df_tolerance=1e-12):
        """
        Check the gradient of the ,odel by comparing to a numerical
        estimate.  If the verbose flag is passed, individual
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

           The *dF_ratio* indicates the limit of numerical accuracy of numerical gradients.
           If it is too small, e.g., smaller than 1e-12, the numerical gradients are usually
           not accurate enough for the tests (shown with blue).
        """
        x = self.optimizer_array.copy()

        if not verbose:
            # make sure only to test the selected parameters
            if target_param is None:
                transformed_index = range(len(x))
            else:
                transformed_index = self._raveled_index_for(target_param)
                if self._has_fixes():
                    indices = np.r_[:self.size]
                    which = (transformed_index[:, None] == indices[self._fixes_][None, :]).nonzero()
                    transformed_index = (indices - (~self._fixes_).cumsum())[transformed_index[which[0]]]

                if transformed_index.size == 0:
                    print("No free parameters to check")
                    return

            # just check the global ratio
            dx = np.zeros(x.shape)
            dx[transformed_index] = step * (np.sign(np.random.uniform(-1, 1, transformed_index.size)) if transformed_index.size != 2 else 1.)

            # evaulate around the point x
            f1 = self._objective(x + dx)
            f2 = self._objective(x - dx)
            gradient = self._grads(x)

            dx = dx[transformed_index]
            gradient = gradient[transformed_index]

            denominator = (2 * np.dot(dx, gradient))
            global_ratio = (f1 - f2) / np.where(denominator == 0., 1e-32, denominator)
            global_diff = np.abs(f1 - f2) < tolerance and np.allclose(gradient, 0, atol=tolerance)
            if global_ratio is np.nan:
                global_ratio = 0
            return np.abs(1. - global_ratio) < tolerance or global_diff
        else:
            # check the gradient of each parameter individually, and do some pretty printing
            try:
                names = self._get_param_names()
            except NotImplementedError:
                names = ['Variable %i' % i for i in range(len(x))]
            # Prepare for pretty-printing
            header = ['Name', 'Ratio', 'Difference', 'Analytical', 'Numerical', 'dF_ratio']
            max_names = max([len(names[i]) for i in range(len(names))] + [len(header[0])])
            float_len = 10
            cols = [max_names]
            cols.extend([max(float_len, len(header[i])) for i in range(1, len(header))])
            cols = np.array(cols) + 5
            header_string = ["{h:^{col}}".format(h=header[i], col=cols[i]) for i in range(len(cols))]
            header_string = list(map(lambda x: '|'.join(x), [header_string]))
            separator = '-' * len(header_string[0])
            print('\n'.join([header_string[0], separator]))
            if target_param is None:
                param_index = range(len(x))
                transformed_index = param_index
            else:
                param_index = self._raveled_index_for(target_param)
                if self._has_fixes():
                    indices = np.r_[:self.size]
                    which = (param_index[:, None] == indices[self._fixes_][None, :]).nonzero()
                    param_index = param_index[which[0]]
                    transformed_index = (indices - (~self._fixes_).cumsum())[param_index]
                    # print param_index, transformed_index
                else:
                    transformed_index = param_index

                if param_index.size == 0:
                    print("No free parameters to check")
                    return

            gradient = self._grads(x).copy()
            np.where(gradient == 0, 1e-312, gradient)
            ret = True
            for nind, xind in zip(param_index, transformed_index):
                xx = x.copy()
                xx[xind] += step
                f1 = self._objective(xx)
                xx[xind] -= 2.*step
                f2 = self._objective(xx)
                #Avoid divide by zero, if any of the values are above 1e-15, otherwise both values are essentiall
                #the same
                if f1 > 1e-15 or f1 < -1e-15 or f2 > 1e-15 or f2 < -1e-15:
                    df_ratio = np.abs((f1 - f2) / min(f1, f2))
                else:
                    df_ratio = 1.0
                df_unstable = df_ratio < df_tolerance
                numerical_gradient = (f1 - f2) / (2 * step)
                if np.all(gradient[xind] == 0): ratio = (f1 - f2) == gradient[xind]
                else: ratio = (f1 - f2) / (2 * step * gradient[xind])
                difference = np.abs(numerical_gradient - gradient[xind])

                if (np.abs(1. - ratio) < tolerance) or np.abs(difference) < tolerance:
                    formatted_name = "\033[92m {0} \033[0m".format(names[nind])
                    ret &= True
                else:
                    formatted_name = "\033[91m {0} \033[0m".format(names[nind])
                    ret &= False
                if df_unstable:
                    formatted_name = "\033[94m {0} \033[0m".format(names[nind])

                r = '%.6f' % float(ratio)
                d = '%.6f' % float(difference)
                g = '%.6f' % gradient[xind]
                ng = '%.6f' % float(numerical_gradient)
                df = '%1.e' % float(df_ratio)
                grad_string = "{0:<{c0}}|{1:^{c1}}|{2:^{c2}}|{3:^{c3}}|{4:^{c4}}|{5:^{c5}}".format(formatted_name, r, d, g, ng, df, c0=cols[0] + 9, c1=cols[1], c2=cols[2], c3=cols[3], c4=cols[4], c5=cols[5])
                print(grad_string)

            self.optimizer_array = x
            return ret

    def _repr_html_(self):
        """Representation of the model in html for notebook display."""
        model_details = [['<b>Model</b>', self.name + '<br>'],
                         ['<b>Log-likelihood</b>', '{}<br>'.format(float(self.log_likelihood()))],
                         ["<b>Number of Parameters</b>", '{}<br>'.format(self.size)],
                         ["<b>Number of Optimization Parameters</b>", '{}<br>'.format(self._size_transformed())],
                         ["<b>Updates</b>", '{}<br>'.format(self._update_on)],
                         ]
        from operator import itemgetter
        to_print = ["""<style type="text/css">
.pd{
    font-family: "Courier New", Courier, monospace !important;
    width: 100%;
    padding: 3px;
}
</style>\n"""] + ["<p class=pd>"] + ["{}: {}".format(name, detail) for name, detail in model_details] + ["</p>"]
        to_print.append(super(Model, self)._repr_html_())
        return "\n".join(to_print)

    def __str__(self):
        model_details = [['Name', self.name],
                         ['Log-likelihood', '{}'.format(float(self.log_likelihood()))],
                         ["Number of Parameters", '{}'.format(self.size)],
                         ["Number of Optimization Parameters", '{}'.format(self._size_transformed())],
                         ["Updates", '{}'.format(self._update_on)],
                         ]
        from operator import itemgetter
        max_len = reduce(lambda a, b: max(len(b[0]), a), model_details, 0)
        to_print = [""] + ["{0:{l}} : {1}".format(name, detail, l=max_len) for name, detail in model_details] + ["Parameters:"]
        to_print.append(super(Model, self).__str__())
        return "\n".join(to_print)

