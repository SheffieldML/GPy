# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
Introduction
^^^^^^^^^^^^

This package principally contains classes ultimately inherited from :py:class:`GPy.core.gp.GP` intended as models for end user consuption - much of :py:class:`GPy.core.gp.GP` is not intended to be called directly. The general form of a "model" is a function that takes some data, a kernel (see :py:class:`GPy.kern`) and other parameters, returning an object representation.

Several models directly inherit :py:class:`GPy.core.gp.GP`:

.. inheritance-diagram:: GPy.models.gp_classification GPy.models.gp_coregionalized_regression GPy.models.gp_heteroscedastic_regression GPy.models.gp_offset_regression GPy.models.gp_regression GPy.models.gp_var_gauss GPy.models.gplvm GPy.models.input_warped_gp GPy.models.multioutput_gp
    :top-classes: GPy.core.gp.GP

Some models fall into conceptually related groups of models (e.g. :py:class:`GPy.core.sparse_gp`, :py:class:`GPy.core.sparse_gp_mpi`):

.. inheritance-diagram:: GPy.models.bayesian_gplvm GPy.models.bayesian_gplvm_minibatch GPy.models.gp_multiout_regression GPy.models.gp_multiout_regression_md GPy.models.ibp_lfm.IBPLFM GPy.models.sparse_gp_coregionalized_regression GPy.models.sparse_gp_minibatch GPy.models.sparse_gp_regression GPy.models.sparse_gp_regression_md GPy.models.sparse_gplvm
    :top-classes: GPy.core.gp.GP

In some cases one end-user model inherits another e.g.

.. inheritance-diagram:: GPy.models.bayesian_gplvm_minibatch
    :top-classes: GPy.models.sparse_gp_minibatch.SparseGPMiniBatch

"""

from .gp_regression import GPRegression
from .gp_classification import GPClassification
from .sparse_gp_regression import SparseGPRegression
from .sparse_gp_classification import SparseGPClassification, SparseGPClassificationUncertainInput
from .gplvm import GPLVM
from .bcgplvm import BCGPLVM
from .sparse_gplvm import SparseGPLVM
from .warped_gp import WarpedGP
from .input_warped_gp import InputWarpedGP
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
from .ibp_lfm import IBPLFM
from .gp_offset_regression import GPOffsetRegression
from .gp_grid_regression import GPRegressionGrid
from .gp_multiout_regression import GPMultioutRegression
from .gp_multiout_regression_md import GPMultioutRegressionMD
from .tp_regression import TPRegression
from .multioutput_gp import MultioutputGP
