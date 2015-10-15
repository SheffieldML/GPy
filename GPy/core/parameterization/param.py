# Copyright (c) 2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from paramz import Param
from .priorizable import Priorizable

class Param(Param, Priorizable):
    pass