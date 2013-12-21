# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import util
import core
import kern
import mappings
import likelihoods
import inference
import models
import examples
import testing
from numpy.testing import Tester
from nose.tools import nottest
from core import priors

@nottest
def tests():
    Tester(testing).test(verbose=10)
