# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import core
from core.parameterization import transformations, priors
constraints = transformations
import models
import mappings
import inference
import util
import examples
import likelihoods
import testing
from numpy.testing import Tester
import kern
import plotting

# Direct imports for convenience:
from core import Model
from core.parameterization import Param, Parameterized, ObsAr

#@nottest
try:
    #Get rid of nose dependency by only ignoring if you have nose installed
    from nose.tools import nottest
    @nottest
    def tests():
        Tester(testing).test(verbose=10)
except:
    def tests():
        Tester(testing).test(verbose=10)

def load(file_path):
    """
    Load a previously pickled model, using `m.pickle('path/to/file.pickle)'

    :param file_name: path/to/file.pickle
    """
    import cPickle as pickle
    try:
        with open(file_path, 'rb') as f:
            m = pickle.load(f)
    except:
        import pickle as pickle
        with open(file_path, 'rb') as f:
            m = pickle.load(f)
    return m
