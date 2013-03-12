import numpy as np
from functools import wraps

def silence_errors(f):
    status = np.seterr(all='ignore')
    @wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)
    np.seterr(**status)
    return wrapper
