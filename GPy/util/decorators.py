import numpy as np
from functools import wraps

def silence_errors(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        status = np.seterr(all='ignore')
        result = f(*args, **kwds)
        np.seterr(**status)
        return result
    return wrapper
