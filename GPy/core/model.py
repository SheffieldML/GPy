# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
from .parameterization.priorizable import Priorizable
from paramz import Model as ParamzModel

class Model(ParamzModel, Priorizable):

    def __init__(self, name):
        super(Model, self).__init__(name)  # Parameterized.__init__(self)

    def _to_dict(self):
        input_dict = {}
        input_dict["name"] = self.name
        return input_dict

    def to_dict(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(input_dict, data=None):
        import copy
        input_dict = copy.deepcopy(input_dict)
        model_class = input_dict.pop('class')
        input_dict["name"] = str(input_dict["name"])
        import GPy
        model_class = eval(model_class)
        return model_class._from_dict(input_dict, data)

    @staticmethod
    def _from_dict(model_class, input_dict, data=None):
        return model_class(**input_dict)

    def save_model(self, output_filename, compress=True, save_data=True):
        raise NotImplementedError

    def _save_model(self, output_filename, compress=True, save_data=True):
        import json
        output_dict = self.to_dict(save_data)
        if compress:
            import gzip
            with gzip.GzipFile(output_filename + ".zip", 'w') as outfile:
                json_str = json.dumps(output_dict)
                json_bytes = json_str.encode('utf-8')
                outfile.write(json_bytes)
        else:
            with open(output_filename + ".json", 'w') as outfile:
                json.dump(output_dict, outfile)

    @staticmethod
    def load_model(output_filename, data=None):
        compress = output_filename.split(".")[-1] == "zip"
        import json
        if compress:
            import gzip
            with gzip.GzipFile(output_filename, 'r') as json_data:
                json_bytes = json_data.read()
                json_str = json_bytes.decode('utf-8')
                output_dict = json.loads(json_str)
        else:
            with open(output_filename) as json_data:
                output_dict = json.load(json_data)
        import GPy
        return GPy.core.model.Model.from_dict(output_dict, data)


    def log_likelihood(self):
        raise NotImplementedError("this needs to be implemented to use the model class")

    def _log_likelihood_gradients(self):
        return self.gradient#.copy()

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
