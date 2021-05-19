'''
Created on Aug 27, 2014

@author: Max Zwiessele
'''
import numpy as np
import warnings

class _Norm(object):
    def __init__(self):
        pass

    def scale_by(self, Y):
        """
        Use data matrix Y as normalization space to work in.
        """
        raise NotImplementedError

    def normalize(self, Y):
        """
        Project Y into normalized space
        """
        if not self.scaled():
            raise AttributeError("Norm object not initialized yet, try calling scale_by(data) first.")

    def inverse_mean(self, X):
        """
        Project the normalized object X into space of Y
        """
        raise NotImplementedError

    def inverse_variance(self, var):
        return var

    def inverse_covariance(self, covariance):
        """
        Convert scaled covariance to unscaled.
        Args:
            covariance - numpy array of shape (n, n)
        Returns:
            covariance - numpy array of shape (n, n, m) where m is number of
                         outputs
        """
        raise NotImplementedError

    def scaled(self):
        """
        Whether this Norm object has been initialized.
        """
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    def _save_to_input_dict(self):
        input_dict = {}
        return input_dict

    @staticmethod
    def from_dict(input_dict):
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
        normalizer_class = input_dict.pop('class')
        import GPy
        normalizer_class = eval(normalizer_class)
        return normalizer_class._build_from_input_dict(normalizer_class, input_dict)

    @staticmethod
    def _build_from_input_dict(normalizer_class, input_dict):
        return normalizer_class(**input_dict)


class Standardize(_Norm):
    def __init__(self):
        self.mean = None

    def scale_by(self, Y):
        Y = np.ma.masked_invalid(Y, copy=False)
        self.mean = Y.mean(0).view(np.ndarray)
        self.std = Y.std(0).view(np.ndarray)
        if np.any(self.std == 0):
            warnings.warn("Some values of Y have standard deviation of zero. Resetting to 1.0 to avoid divide by zero errors.")
            self.std[np.where(self.std==0)]=1.

    def normalize(self, Y):
        super(Standardize, self).normalize(Y)
        return (Y-self.mean)/self.std

    def inverse_mean(self, X):
        return (X*self.std)+self.mean

    def inverse_variance(self, var):
        return (var*(self.std**2))

    def inverse_covariance(self, covariance):
        return (covariance[..., np.newaxis]*(self.std**2))

    def scaled(self):
        return self.mean is not None

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Standardize, self)._save_to_input_dict()
        input_dict["class"] = "GPy.util.normalizer.Standardize"
        if self.mean is not None:
            input_dict["mean"] = self.mean.tolist()
            input_dict["std"] = self.std.tolist()
        return input_dict

    @staticmethod
    def _build_from_input_dict(kernel_class, input_dict):
        s = Standardize()
        if "mean" in input_dict:
            s.mean = np.array(input_dict["mean"])
        if "std" in input_dict:
            s.std = np.array(input_dict["std"])
        return s
