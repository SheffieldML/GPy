'''
.. module:: GPy.models

Implementations for common models used in GP regression and classification.
The different models can be viewed in :mod:`GPy.models_modules`, which holds
detailed explanations for the different models.

.. note::
    This module is a convienince module for endusers to use. For developers 
    see :mod:`GPy.models_modules`, which holds the implementions for each model.: 

.. moduleauthor:: Max Zwiessele <ibinbei@gmail.com>
'''

__updated__ = '2013-11-28'

from models_modules.bayesian_gplvm import BayesianGPLVM, BayesianGPLVMWithMissingData
from models_modules.gp_regression import GPRegression
from models_modules.gp_classification import GPClassification#; _gp_classification = gp_classification ; del gp_classification 
from models_modules.sparse_gp_regression import SparseGPRegression#; _sparse_gp_regression = sparse_gp_regression ; del sparse_gp_regression 
from models_modules.svigp_regression import SVIGPRegression#; _svigp_regression = svigp_regression ; del svigp_regression 
from models_modules.sparse_gp_classification import SparseGPClassification#; _sparse_gp_classification = sparse_gp_classification ; del sparse_gp_classification 
from models_modules.fitc_classification import FITCClassification#; _fitc_classification = fitc_classification ; del fitc_classification 
from models_modules.gplvm import GPLVM#; _gplvm = gplvm ; del gplvm 
from models_modules.bcgplvm import BCGPLVM#; _bcgplvm = bcgplvm; del bcgplvm
from models_modules.sparse_gplvm import SparseGPLVM#; _sparse_gplvm = sparse_gplvm ; del sparse_gplvm 
from models_modules.warped_gp import WarpedGP#; _warped_gp = warped_gp ; del warped_gp 
from models_modules.bayesian_gplvm import BayesianGPLVM#; _bayesian_gplvm = bayesian_gplvm ; del bayesian_gplvm 
from models_modules.mrd import MRD#; _mrd = mrd; del mrd 
from models_modules.gradient_checker import GradientChecker#; _gradient_checker = gradient_checker ; del gradient_checker 
from models_modules.gp_multioutput_regression import GPMultioutputRegression#; _gp_multioutput_regression = gp_multioutput_regression ; del gp_multioutput_regression 
from models_modules.sparse_gp_multioutput_regression import SparseGPMultioutputRegression#; _sparse_gp_multioutput_regression = sparse_gp_multioutput_regression ; del sparse_gp_multioutput_regression 
from models_modules.gradient_checker import GradientChecker