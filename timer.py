from functools import wraps
import time
import cupy as cp

def stopwatch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        cp.cuda.Stream.null.synchronize()
        start = time.time()
        
        result = func(*args, **kwargs)
        
        cp.cuda.Stream.null.synchronize()
        end = time.time()
        return end-start, result
    return wrapper
