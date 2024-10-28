# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import pytest
import numpy as np
import GPy
from GPy.core.parameterization.priors import (
    Gaussian,
    Uniform,
    LogGaussian,
    MultivariateGaussian,
    Gamma,
    InverseGamma,
    DGPLVM,
    DGPLVM_KFDA,
    DGPLVM_Lamda,
    DGPLVM_T,
    HalfT,
    Exponential,
    StudentT,
)


class TestPrior:
    def test_studentT(self):
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        studentT = GPy.priors.StudentT(1, 2, 4)

        m = GPy.models.SparseGPRegression(X, y)
        m.Z.set_prior(studentT)

        # setting a StudentT prior on non-negative parameters
        # should raise an assertionerror.

        with pytest.raises(AssertionError):
            m.rbf.set_prior(studentT)

        # The gradients need to be checked
        assert m.checkgrad()

        # Check the singleton pattern:
        assert studentT is GPy.priors.StudentT(1, 2, 4)
        assert studentT is not GPy.priors.StudentT(2, 2, 4)

    def test_lognormal(self):
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)
        lognormal = GPy.priors.LogGaussian(1, 2)
        m.rbf.set_prior(lognormal)
        m.randomize()
        assert m.checkgrad()

    def test_Gamma(self):
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)
        Gamma = GPy.priors.Gamma(1, 1)
        m.rbf.set_prior(Gamma)
        m.randomize()
        assert m.checkgrad()

    def test_InverseGamma(self):
        # Test that this prior object can be instantiated and performs its basic functions
        # in integration.
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)
        InverseGamma = GPy.priors.InverseGamma(1, 1)
        m.rbf.set_prior(InverseGamma)
        m.randomize()
        assert m.checkgrad()

    def test_incompatibility(self):
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)
        gaussian = GPy.priors.Gaussian(1, 1)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        with pytest.raises(AssertionError):
            m.rbf.set_prior(gaussian)

    def test_set_prior(self):
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)

        gaussian = GPy.priors.Gaussian(1, 1)
        # m.rbf.set_prior(gaussian)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        with pytest.raises(AssertionError):
            m.rbf.set_prior(gaussian)

    def test_uniform(self):
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.SparseGPRegression(X, y)
        uniform = GPy.priors.Uniform(0, 2)
        m.rbf.set_prior(uniform)
        m.randomize()
        assert m.checkgrad()

        m.Z.set_prior(uniform)
        m.randomize()
        assert m.checkgrad()

        m.Z.unconstrain()
        uniform = GPy.priors.Uniform(-1, 10)
        m.Z.set_prior(uniform)
        m.randomize()
        assert m.checkgrad()

        m.Z.constrain_negative()
        uniform = GPy.priors.Uniform(-1, 0)
        m.Z.set_prior(uniform)
        m.randomize()
        assert m.checkgrad()

    def test_set_gaussian_for_reals(self):
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.SparseGPRegression(X, y)

        gaussian = GPy.priors.Gaussian(1, 1)
        m.Z.set_prior(gaussian)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        # self.assertRaises(AssertionError, m.Z.set_prior, gaussian)
        assert m.checkgrad()

    def test_fixed_domain_check(self):
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)

        m.rbf.fix()
        gaussian = GPy.priors.Gaussian(1, 1)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        with pytest.raises(AssertionError):
            m.rbf.set_prior(gaussian)

    def test_fixed_domain_check1(self):
        xmin, xmax = 1, 2.5 * np.pi
        b, C, SNR = 1, 0, 0.1
        X = np.linspace(xmin, xmax, 500)
        y = b * X + C + 1 * np.sin(X)
        y += 0.05 * np.random.randn(len(X))
        X, y = X[:, None], y[:, None]
        m = GPy.models.GPRegression(X, y)

        m.kern.lengthscale.fix()
        gaussian = GPy.priors.Gaussian(1, 1)
        # setting a Gaussian prior on non-negative parameters
        # should raise an assertionerror.
        with pytest.raises(AssertionError):
            m.rbf.set_prior(gaussian)


def initialize_gaussian_prior() -> None:
    return Gaussian(0, 1)


def initialize_uniform_prior() -> None:
    return Uniform(0, 1)


def initialize_log_gaussian_prior() -> None:
    return LogGaussian(0, 1)


def initialize_multivariate_gaussian_prior() -> None:
    return MultivariateGaussian(np.zeros(2), np.eye(2))


def initialize_gamma_prior() -> None:
    return Gamma(1, 1)


def initialize_inverse_gamma_prior() -> None:
    return InverseGamma(1, 1)


def initialize_dgplvm_prior() -> None:
    # return DGPLVM(...)
    raise NotImplementedError("No idea how to initialize this prior")


def initialize_dgplvm_kfda_prior() -> None:
    # return DGPLVM_KFDA(...)
    raise NotImplementedError("No idea how to initialize this prior")


def initialize_dgplvm_lamda_prior() -> None:
    # return DGPLVM_Lamda(...)
    raise NotImplementedError("No idea how to initialize this prior")


def initialize_dgplvm_t_prior() -> None:
    # return DGPLVM_T(1, 1, (1, 1))
    raise NotImplementedError("No idea how to initialize this prior")


def initialize_half_t_prior() -> None:
    return HalfT(1, 1)


def initialize_exponential_prior() -> None:
    return Exponential(1)


def initialize_student_t_prior() -> None:
    return StudentT(1, 1, 1)


PRIORS = {
    "Gaussian": initialize_gaussian_prior,
    "Uniform": initialize_uniform_prior,
    "LogGaussian": initialize_log_gaussian_prior,
    "MultivariateGaussian": initialize_multivariate_gaussian_prior,
    "Gamma": initialize_gamma_prior,
    "InverseGamma": initialize_inverse_gamma_prior,
    # "DGPLVM": initialize_dgplvm_prior,
    # "DGPLVM_KFDA": initialize_dgplvm_kfda_prior,
    # "DGPLVM_Lamda": initialize_dgplvm_lamda_prior,
    # "DGPLVM_T": initialize_dgplvm_t_prior,
    "HalfT": initialize_half_t_prior,
    "Exponential": initialize_exponential_prior,
    "StudentT": initialize_student_t_prior,
}


def check_prior(prior_getter: str) -> None:
    prior_getter()


def test_priors() -> None:
    for prior_name, prior_getter in PRIORS.items():
        try:
            check_prior(prior_getter)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {prior_name} prior"
            ) from e  # noqa E501
