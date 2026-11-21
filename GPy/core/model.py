# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
from .parameterization.priorizable import Priorizable
from .parameterization import ParamzModel
import multiprocessing as mp
import numpy as np


def _opt_wrapper(args):
    model, kwargs = args
    return model.optimize(**kwargs)


class Model(ParamzModel, Priorizable):
    def __init__(self, name):
        super(Model, self).__init__(name)  # Parameterized.__init__(self)

    def optimize(
        self,
        optimizer=None,
        start=None,
        messages=False,
        max_iters=1000,
        ipython_notebook=True,
        clear_after_finish=False,
        **kwargs,
    ):
        """
        Optimize the model using self.log_likelihood and self.log_likelihood_gradient,
        as well as self.priors.

        CIP-000F Stage 1: dispatch uses GPy-owned ``GPy.inference.optimization``
        wrappers rather than ``paramz.optimization``.

        :param max_iters: maximum number of function evaluations
        :type max_iters: int
        :param messages: whether to display during optimisation
        :type messages: bool
        :param optimizer: which optimizer to use (defaults to self.preferred_optimizer)
        :type optimizer: string
        """
        from GPy.inference import optimization
        from GPy.inference.optimization.verbose_optimization import VerboseOptimization

        if self.is_fixed or self.size == 0:
            print("nothing to optimize")
            return

        if not self.update_model():
            print("updates were off, setting updates on again")
            self.update_model(True)

        if start is None:
            start = self.optimizer_array

        if optimizer is None:
            optimizer = self.preferred_optimizer

        if isinstance(optimizer, optimization.Optimizer):
            opt = optimizer
            opt.model = self
        else:
            optimizer = optimization.get_optimizer(optimizer)
            opt = optimizer(max_iters=max_iters, **kwargs)

        with VerboseOptimization(
            self,
            opt,
            maxiters=max_iters,
            verbose=messages,
            ipython_notebook=ipython_notebook,
            clear_after_finish=clear_after_finish,
        ):
            opt.run(start, f_fp=self._objective_grads, f=self._objective, fp=self._grads)

        self.optimizer_array = opt.x_opt
        self.optimization_runs.append(opt)
        return opt

    def optimize_restarts(
        self,
        num_restarts=10,
        robust=False,
        verbose=True,
        parallel=False,
        num_processes=None,
        **kwargs,
    ):
        r"""
        Perform random restarts of the model, and set the model to the best
        seen solution.

        If the robust flag is set, exceptions raised during optimizations will
        be handled silently.  If _all_ runs fail, the model is reset to the
        existing parameter values.

        \*\*kwargs are passed to the optimizer.

        :param num_restarts: number of restarts to use (default 10)
        :type num_restarts: int
        :param robust: whether to handle exceptions silently or not (default False)
        :type robust: bool
        :param parallel: whether to run each restart as a separate process
        :type parallel: bool
        :param num_processes: number of workers in the multiprocessing pool
        :type num_processes: int
        """
        initial_length = len(self.optimization_runs)
        initial_parameters = self.optimizer_array.copy()

        if parallel:  # pragma: no cover
            try:
                pool = mp.Pool(processes=num_processes)
                obs = [self.copy() for i in range(num_restarts)]
                [obs[i].randomize() for i in range(num_restarts - 1)]
                jobs = pool.map(_opt_wrapper, [(o, kwargs) for o in obs])
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                print("Ctrl+c received, terminating and joining pool.")
                pool.terminate()
                pool.join()

        for i in range(num_restarts):
            try:
                if not parallel:
                    if i > 0:
                        self.randomize()
                    self.optimize(**kwargs)
                else:  # pragma: no cover
                    self.optimization_runs.append(jobs[i])

                if verbose:
                    print(f"Optimization restart {i + 1}/{num_restarts}, f = {self.optimization_runs[-1].f_opt}")
            except Exception as e:
                if robust:
                    print(f"Warning - optimization restart {i + 1}/{num_restarts} failed")
                else:
                    raise e

        if len(self.optimization_runs) > initial_length:
            i = np.argmin([o.f_opt for o in self.optimization_runs[initial_length:]])
            self.optimizer_array = self.optimization_runs[initial_length + i].x_opt
        else:
            self.optimizer_array = initial_parameters
        return self.optimization_runs

    def _save_to_input_dict(self):
        """
        It is used by the public method to_dict to create json serializable dictionary.
        """
        input_dict = {}
        input_dict["name"] = self.name
        return input_dict

    def to_dict(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(input_dict, data=None):
        """
        Instantiate an object of a derived class using the information
        in input_dict (built by the to_dict method of the derived class).
        More specifically, after reading the derived class from input_dict,
        it calls the method _build_from_input_dict of the derived class.
        Note: This method should not be overrided in the derived class. In case
        it is needed, please override _build_from_input_dict instate.

        :param dict input_dict: Dictionary with all the information needed to
           instantiate the object.
        """
        import copy

        input_dict = copy.deepcopy(input_dict)
        model_class = input_dict.pop("class")
        input_dict["name"] = str(input_dict["name"])
        import GPy  # noqa: F401 — needed for eval of serialized class paths

        model_class = eval(model_class)
        return model_class._build_from_input_dict(input_dict, data)

    @staticmethod
    def _build_from_input_dict(model_class, input_dict, data=None):
        """
        This method is used by the public method from_dict to build an object
        of class model_class using the information contained in input_dict.
        Note: This method is often overrided in the derived class to deal with
        any pre-processing of the parameters in input_dict before calling the
        constructor of the object.

        :param str model_class: Class of the object to build.
        :param dict input_dict: Extra information needed by the constructor of model_class.
        """
        return model_class(**input_dict)

    def save_model(self, output_filename, compress=True, save_data=True):
        raise NotImplementedError

    def _save_model(self, output_filename, compress=True, save_data=True):
        import json

        output_dict = self.to_dict(save_data)
        if compress:
            import gzip

            with gzip.GzipFile(output_filename + ".zip", "w") as outfile:
                json_str = json.dumps(output_dict)
                json_bytes = json_str.encode("utf-8")
                outfile.write(json_bytes)
        else:
            with open(output_filename + ".json", "w") as outfile:
                json.dump(output_dict, outfile)

    @staticmethod
    def load_model(output_filename, data=None):
        compress = output_filename.split(".")[-1] == "zip"
        import json

        if compress:
            import gzip

            with gzip.GzipFile(output_filename, "r") as json_data:
                json_bytes = json_data.read()
                json_str = json_bytes.decode("utf-8")
                output_dict = json.loads(json_str)
        else:
            with open(output_filename) as json_data:
                output_dict = json.load(json_data)
        import GPy

        return GPy.core.model.Model.from_dict(output_dict, data)

    def log_likelihood(self):
        raise NotImplementedError("this needs to be implemented to use the model class")

    def _log_likelihood_gradients(self):
        return self.gradient  # .copy()

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
        return -np.asarray(self.log_likelihood()).item() - self.log_prior()

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
