# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from .. import likelihoods
from ..inference import optimization
from ..util.linalg import jitchol
from GPy.util.misc import opt_wrapper
from parameterised import Parameterised
import multiprocessing as mp
import numpy as np
from GPy.core.domains import POSITIVE, REAL
from numpy.linalg.linalg import LinAlgError
# import numdifftools as ndt

class Model(Parameterised):
    _fail_count = 0 # Count of failed optimization steps (see objective)
    _allowed_failures = 10 # number of allowed failures
    def __init__(self):
        Parameterised.__init__(self)
        self.priors = None
        self.optimization_runs = []
        self.sampling_runs = []
        self.preferred_optimizer = 'scg'
        # self._set_params(self._get_params()) has been taken out as it should only be called on leaf nodes
    def _get_params(self):
        raise NotImplementedError, "this needs to be implemented to use the Model class"
    def _set_params(self, x):
        raise NotImplementedError, "this needs to be implemented to use the Model class"
    def log_likelihood(self):
        raise NotImplementedError, "this needs to be implemented to use the Model class"
    def _log_likelihood_gradients(self):
        raise NotImplementedError, "this needs to be implemented to use the Model class"

    def set_prior(self, regexp, what):
        """
        Sets priors on the Model parameters.

        Arguments
        ---------
        regexp -- string, regexp, or integer array
        what -- instance of a Prior class

        Notes
        -----
        Asserts that the Prior is suitable for the constraint. If the
        wrong constraint is in place, an error is raised.  If no
        constraint is in place, one is added (warning printed).

        For tied parameters, the Prior will only be "counted" once, thus
        a Prior object is only inserted on the first tied index
        """
        if self.priors is None:
            self.priors = [None for i in range(self._get_params().size)]

        which = self.grep_param_names(regexp)

        # check tied situation
        tie_partial_matches = [tie for tie in self.tied_indices if (not set(tie).isdisjoint(set(which))) & (not set(tie) == set(which))]
        if len(tie_partial_matches):
            raise ValueError, "cannot place Prior across partial ties"
        tie_matches = [tie for tie in self.tied_indices if set(which) == set(tie) ]
        if len(tie_matches) > 1:
            raise ValueError, "cannot place Prior across multiple ties"
        elif len(tie_matches) == 1:
            which = which[:1] # just place a Prior object on the first parameter


        # check constraints are okay

        if what.domain is POSITIVE:
            constrained_positive_indices = [i for i, t in zip(self.constrained_indices, self.constraints) if t.domain is POSITIVE]
            if len(constrained_positive_indices):
                constrained_positive_indices = np.hstack(constrained_positive_indices)
            else:
                constrained_positive_indices = np.zeros(shape=(0,))
            bad_constraints = np.setdiff1d(self.all_constrained_indices(), constrained_positive_indices)
            assert not np.any(which[:, None] == bad_constraints), "constraint and Prior incompatible"
            unconst = np.setdiff1d(which, constrained_positive_indices)
            if len(unconst):
                print "Warning: constraining parameters to be positive:"
                print '\n'.join([n for i, n in enumerate(self._get_param_names()) if i in unconst])
                print '\n'
                self.constrain_positive(unconst)
        elif what.domain is REAL:
            assert not np.any(which[:, None] == self.all_constrained_indices()), "constraint and Prior incompatible"
        else:
            raise ValueError, "Prior not recognised"

        # store the Prior in a local list
        for w in which:
            self.priors[w] = what

    def get_gradient(self, name, return_names=False):
        """
        Get Model gradient(s) by name. The name is applied as a regular expression and all parameters that match that regular expression are returned.
        """
        matches = self.grep_param_names(name)
        if len(matches):
            if return_names:
                return self._log_likelihood_gradients()[matches], np.asarray(self._get_param_names())[matches].tolist()
            else:
                return self._log_likelihood_gradients()[matches]
        else:
            raise AttributeError, "no parameter matches %s" % name

    def log_prior(self):
        """evaluate the prior"""
        if self.priors is not None:
            return np.sum([p.lnpdf(x) for p, x in zip(self.priors, self._get_params()) if p is not None])
        else:
            return 0.

    def _log_prior_gradients(self):
        """evaluate the gradients of the priors"""
        if self.priors is None:
            return 0.
        x = self._get_params()
        ret = np.zeros(x.size)
        [np.put(ret, i, p.lnpdf_grad(xx)) for i, (p, xx) in enumerate(zip(self.priors, x)) if not p is None]
        return ret

    def _transform_gradients(self, g):
        x = self._get_params()
        for index, constraint in zip(self.constrained_indices, self.constraints):
            g[index] = g[index] * constraint.gradfactor(x[index])
        [np.put(g, i, v) for i, v in [(t[0], np.sum(g[t])) for t in self.tied_indices]]
        if len(self.tied_indices) or len(self.fixed_indices):
            to_remove = np.hstack((self.fixed_indices + [t[1:] for t in self.tied_indices]))
            return np.delete(g, to_remove)
        else:
            return g

    def randomize(self):
        """
        Randomize the Model.
        Make this draw from the Prior if one exists, else draw from N(0,1)
        """
        # first take care of all parameters (from N(0,1))
        x = self._get_params_transformed()
        x = np.random.randn(x.size)
        self._set_params_transformed(x)
        # now draw from Prior where possible
        x = self._get_params()
        if self.priors is not None:
            [np.put(x, i, p.rvs(1)) for i, p in enumerate(self.priors) if not p is None]
        self._set_params(x)
        self._set_params_transformed(self._get_params_transformed()) # makes sure all of the tied parameters get the same init (since there's only one prior object...)


    def optimize_restarts(self, num_restarts=10, robust=False, verbose=True, parallel=False, num_processes=None, **kwargs):
        """
        Perform random restarts of the Model, and set the Model to the best
        seen solution.

        If the robust flag is set, exceptions raised during optimizations will
        be handled silently.  If _all_ runs fail, the Model is reset to the
        existing parameter values.

        Notes
        -----
        **kwargs are passed to the optimizer. They can be:
        :max_f_eval: maximum number of function evaluations
        :messages: whether to display during optimisation
        :verbose: whether to show informations about the current restart
        :parallel: whether to run each restart as a separate process. It relies on the multiprocessing module.
        :num_processes: number of workers in the multiprocessing pool

        ..Note: If num_processes is None, the number of workes in the multiprocessing pool is automatically
        set to the number of processors on the current machine.


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

                pool.close() # signal that no more data coming in
                pool.join() # wait for all the tasks to complete
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

    def ensure_default_constraints(self):
        """
        Ensure that any variables which should clearly be positive have been constrained somehow.
        """
        positive_strings = ['variance', 'lengthscale', 'precision', 'kappa']
        # param_names = self._get_param_names()
        currently_constrained = self.all_constrained_indices()
        to_make_positive = []
        for s in positive_strings:
            for i in self.grep_param_names(".*" + s):
                if not (i in currently_constrained):
                    to_make_positive.append(i)
        if len(to_make_positive):
            self.constrain_positive(np.asarray(to_make_positive))

    def objective_function(self, x):
        """
        The objective function passed to the optimizer. It combines the likelihood and the priors.
        
        Failures are handled robustly. The algorithm will try several times to
        return the objective, and will raise the original exception if it
        the objective cannot be computed.
        """
        try:
            self._set_params_transformed(x)
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError) as e:
            if self._fail_count >= self._allowed_failures:
                raise e
            self._fail_count += 1
            return np.inf
        return -self.log_likelihood() - self.log_prior()

    def objective_function_gradients(self, x):
        """
        Gets the gradients from the likelihood and the priors.

        Failures are handled robustly. The algorithm will try several times to
        return the objective, and will raise the original exception if it
        the objective cannot be computed.
        """
        try:
            self._set_params_transformed(x)
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError) as e:
            if self._fail_count >= self._allowed_failures:
                raise e
            self._fail_count += 1
        obj_grads = -self._transform_gradients(self._log_likelihood_gradients() + self._log_prior_gradients())
        return obj_grads

    def objective_and_gradients(self, x):
        try:
            self._set_params_transformed(x)
            obj_f = -self.log_likelihood() - self.log_prior()
            self._fail_count = 0
        except (LinAlgError, ZeroDivisionError, ValueError) as e:
            if self._fail_count >= self._allowed_failures:
                raise e
            self._fail_count += 1
            obj_f = np.inf
        obj_grads = -self._transform_gradients(self._log_likelihood_gradients() + self._log_prior_gradients())
        return obj_f, obj_grads

    def optimize(self, optimizer=None, start=None, **kwargs):
        """
        Optimize the Model using self.log_likelihood and self.log_likelihood_gradient, as well as self.priors.
        kwargs are passed to the optimizer. They can be:

        :max_f_eval: maximum number of function evaluations
        :messages: whether to display during optimisation
        :param optimzer: which optimizer to use (defaults to self.preferred optimizer)
        :type optimzer: string TODO: valid strings?
        """
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
        sgd = SGD.StochasticGD(self, iterations, learning_rate, momentum, **kwargs) # @UndefinedVariable
        sgd.run()
        self.optimization_runs.append(sgd)

    def Laplace_covariance(self):
        """return the covariance matric of a Laplace approximatino at the current (stationary) point"""
        # TODO add in the Prior contributions for MAP estimation
        # TODO fix the hessian for tied, constrained and fixed components
        if hasattr(self, 'log_likelihood_hessian'):
            A = -self.log_likelihood_hessian()

        else:
            print "numerically calculating hessian. please be patient!"
            x = self._get_params()
            def f(x):
                self._set_params(x)
                return self.log_likelihood()
            h = ndt.Hessian(f) # @UndefinedVariable
            A = -h(x)
            self._set_params(x)
        # check for almost zero components on the diagonal which screw up the cholesky
        aa = np.nonzero((np.diag(A) < 1e-6) & (np.diag(A) > 0.))[0]
        A[aa, aa] = 0.
        return A

    def Laplace_evidence(self):
        """Returns an estiamte of the Model evidence based on the Laplace approximation.
        Uses a numerical estimate of the hessian if none is available analytically"""
        A = self.Laplace_covariance()
        try:
            hld = np.sum(np.log(np.diag(jitchol(A)[0])))
        except:
            return np.nan
        return 0.5 * self._get_params().size * np.log(2 * np.pi) + self.log_likelihood() - hld

    def __str__(self):
        s = Parameterised.__str__(self).split('\n')
        # add priors to the string
        if self.priors is not None:
            strs = [str(p) if p is not None else '' for p in self.priors]
        else:
            strs = [''] * len(self._get_params())
        width = np.array(max([len(p) for p in strs] + [5])) + 4

        log_like = self.log_likelihood()
        log_prior = self.log_prior()
        obj_funct = '\nLog-likelihood: {0:.3e}'.format(log_like)
        if len(''.join(strs)) != 0:
            obj_funct += ', Log Prior: {0:.3e}, LL+Prior = {0:.3e}'.format(log_prior, log_like + log_prior)
        obj_funct += '\n\n'
        s[0] = obj_funct + s[0]
        s[0] += "|{h:^{col}}".format(h='Prior', col=width)
        s[1] += '-' * (width + 1)

        for p in range(2, len(strs) + 2):
            s[p] += '|{Prior:^{width}}'.format(Prior=strs[p - 2], width=width)

        return '\n'.join(s)


    def checkgrad(self, target_param=None, verbose=False, step=1e-6, tolerance=1e-3):
        """
        Check the gradient of the Model by comparing to a numerical estimate.
        If the verbose flag is passed, invividual components are tested (and printed)

        :param verbose: If True, print a "full" checking of each parameter
        :type verbose: bool
        :param step: The size of the step around which to linearise the objective
        :type step: float (defaul 1e-6)
        :param tolerance: the tolerance allowed (see note)
        :type tolerance: float (default 1e-3)

        Note:-
           The gradient is considered correct if the ratio of the analytical
           and numerical gradients is within <tolerance> of unity.
        """

        x = self._get_params_transformed().copy()

        if not verbose:
            # just check the global ratio
            dx = step * np.sign(np.random.uniform(-1, 1, x.size))

            # evaulate around the point x
            f1, g1 = self.objective_and_gradients(x + dx)
            f2, g2 = self.objective_and_gradients(x - dx)
            gradient = self.objective_function_gradients(x)

            numerical_gradient = (f1 - f2) / (2 * dx)
            global_ratio = (f1 - f2) / (2 * np.dot(dx, gradient))

            return (np.abs(1. - global_ratio) < tolerance) or (np.abs(gradient - numerical_gradient).mean() - 1) < tolerance
        else:
            # check the gradient of each parameter individually, and do some pretty printing
            try:
                names = self._get_param_names_transformed()
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
                param_list = range(len(x))
            else:
                param_list = self.grep_param_names(target_param, transformed=True, search=True)
                if not np.any(param_list):
                    print "No free parameters to check"
                    return


            for i in param_list:
                xx = x.copy()
                xx[i] += step
                f1, g1 = self.objective_and_gradients(xx)
                xx[i] -= 2.*step
                f2, g2 = self.objective_and_gradients(xx)
                gradient = self.objective_function_gradients(x)[i]

                numerical_gradient = (f1 - f2) / (2 * step)
                ratio = (f1 - f2) / (2 * step * gradient)
                difference = np.abs((f1 - f2) / 2 / step - gradient)

                if (np.abs(1. - ratio) < tolerance) or np.abs(difference) < tolerance:
                    formatted_name = "\033[92m {0} \033[0m".format(names[i])
                else:
                    formatted_name = "\033[91m {0} \033[0m".format(names[i])
                r = '%.6f' % float(ratio)
                d = '%.6f' % float(difference)
                g = '%.6f' % gradient
                ng = '%.6f' % float(numerical_gradient)
                grad_string = "{0:^{c0}}|{1:^{c1}}|{2:^{c2}}|{3:^{c3}}|{4:^{c4}}".format(formatted_name, r, d, g, ng, c0=cols[0] + 9, c1=cols[1], c2=cols[2], c3=cols[3], c4=cols[4])
                print grad_string

    def input_sensitivity(self):
        """
        return an array describing the sesitivity of the Model to each input

        NB. Right now, we're basing this on the lengthscales (or
        variances) of the kernel.  TODO: proper sensitivity analysis
        where we integrate across the Model inputs and evaluate the
        effect on the variance of the Model output.  """

        if not hasattr(self, 'kern'):
            raise ValueError, "this Model has no kernel"

        k = [p for p in self.kern.parts if p.name in ['rbf', 'linear']]
        if (not len(k) == 1) or (not k[0].ARD):
            raise ValueError, "cannot determine sensitivity for this kernel"
        k = k[0]

        if k.name == 'rbf':
            return k.lengthscale
        elif k.name == 'linear':
            return 1. / k.variances


    def pseudo_EM(self, epsilon=.1, **kwargs):
        """
        TODO: Should this not bein the GP class?
        EM - like algorithm  for Expectation Propagation and Laplace approximation

        kwargs are passed to the optimize function. They can be:

        :epsilon: convergence criterion
        :max_f_eval: maximum number of function evaluations
        :messages: whether to display during optimisation
        :param optimzer: whice optimizer to use (defaults to self.preferred optimizer)
        :type optimzer: string TODO: valid strings?

        """
        assert isinstance(self.likelihood, likelihoods.EP), "pseudo_EM is only available for EP likelihoods"
        ll_change = epsilon + 1.
        iteration = 0
        last_ll = -np.inf

        convergence = False
        alpha = 0
        stop = False

        while not stop:
            last_approximation = self.likelihood.copy()
            last_params = self._get_params()
            self.update_likelihood_approximation()
            new_ll = self.log_likelihood()
            ll_change = new_ll - last_ll

            if ll_change < 0:
                self.likelihood = last_approximation # restore previous likelihood approximation
                self._set_params(last_params) # restore model parameters
                print "Log-likelihood decrement: %s \nLast likelihood update discarded." % ll_change
                stop = True
            else:
                self.optimize(**kwargs)
                last_ll = self.log_likelihood()
                if ll_change < epsilon:
                    stop = True
            iteration += 1
            if stop:
                print "%s iterations." % iteration
        self.update_likelihood_approximation()
