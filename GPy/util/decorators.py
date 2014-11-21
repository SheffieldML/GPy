# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from functools import wraps

def silence_errors(f):
    """
    This wraps a function and it silences numpy errors that
    happen during the execution. After the function has exited, it restores
    the previous state of the warnings.
    """
    @wraps(f)
    def wrapper(*args, **kwds):
        status = np.seterr(all='ignore')
        result = f(*args, **kwds)
        np.seterr(**status)
        return result
    return wrapper
