# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import core
import models
import inference
import util
import examples
import likelihoods
import testing
from numpy.testing import Tester
from nose.tools import nottest
import kern
from core import priors

@nottest
def tests():
    Tester(testing).test(verbose=10)
