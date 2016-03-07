# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from paramz import Param
from .priorizable import Priorizable
from paramz.transformations import __fixed__
import logging, numpy as np

class Param(Param, Priorizable):
    pass
