# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .gp_regression import GPRegression
from .gp_classification import GPClassification
from .sparse_gp_regression import SparseGPRegression
from .sparse_gp_classification import SparseGPClassification, SparseGPClassificationUncertainInput
from .gplvm import GPLVM
from .bcgplvm import BCGPLVM
from .sparse_gplvm import SparseGPLVM
from .warped_gp import WarpedGP
from .bayesian_gplvm import BayesianGPLVM
from .mrd import MRD
from .gradient_checker import GradientChecker, HessianChecker, SkewChecker
from .ss_gplvm import SSGPLVM
from .gp_coregionalized_regression import GPCoregionalizedRegression
from .sparse_gp_coregionalized_regression import SparseGPCoregionalizedRegression
from .gp_heteroscedastic_regression import GPHeteroscedasticRegression
from .ss_mrd import SSMRD
from .gp_kronecker_gaussian_regression import GPKroneckerGaussianRegression
from .gp_var_gauss import GPVariationalGaussianApproximation
from .one_vs_all_classification import OneVsAllClassification
from .one_vs_all_sparse_classification import OneVsAllSparseClassification
from .dpgplvm import DPBayesianGPLVM
from .state_space_model import StateSpace
from .gp_grid_regression import GPRegressionGrid
