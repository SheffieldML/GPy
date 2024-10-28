import numpy
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


def initialize_gaussian_prior() -> None:
    return Gaussian(0, 1)


def initialize_uniform_prior() -> None:
    return Uniform(0, 1)


def initialize_log_gaussian_prior() -> None:
    return LogGaussian(0, 1)


def initialize_multivariate_gaussian_prior() -> None:
    return MultivariateGaussian(numpy.zeros(2), numpy.eye(2))


def initialize_gamma_prior() -> None:
    return Gamma(1, 1)


def initialize_inverse_gamma_prior() -> None:
    return InverseGamma(1, 1)


def initialize_dgplvm_prior() -> None:
    return DGPLVM(1, 1, (1, 1))


def initialize_dgplvm_kfda_prior() -> None:
    return DGPLVM_KFDA(1, 1, (1, 1))


def initialize_dgplvm_lamda_prior() -> None:
    return DGPLVM_Lamda(1, 1, (1, 1))


def initialize_dgplvm_t_prior() -> None:
    return DGPLVM_T(1, 1, (1, 1))


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
    "DGPLVM": initialize_dgplvm_prior,
    "DGPLVM_KFDA": initialize_dgplvm_kfda_prior,
    "DGPLVM_Lamda": initialize_dgplvm_lamda_prior,
    "DGPLVM_T": initialize_dgplvm_t_prior,
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
