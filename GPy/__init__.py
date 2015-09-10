# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from . import core
from .core.parameterization import transformations, priors
constraints = transformations
from . import models
from . import mappings
from . import inference
from . import util
from . import examples
from . import likelihoods
from . import testing
from numpy.testing import Tester
from . import kern
from . import plotting

# Direct imports for convenience:
from .core import Model
from .core.parameterization import Param, Parameterized, ObsAr

from .__version__ import __version__

#@nottest
try:
    #Get rid of nose dependency by only ignoring if you have nose installed
    from nose.tools import nottest
    @nottest
    def tests(verbose=10):
        Tester(testing).test(verbose=verbose)
except:
    def tests(verbose=10):
        Tester(testing).test(verbose=verbose)

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
