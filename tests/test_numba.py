import os
os.environ["NUMBA_NUM_THREADS"] = "8"

import numpy as np
import time
from numba import njit, prange, config

print(f"Numba threads available: {config.NUMBA_NUM_THREADS}")

@njit(parallel=True, cache=False)
def parallel_test(n):
    result = np.zeros(n)
    for i in prange(n):
        result[i] = i ** 0.5
    return result

# Warmup
parallel_test(100)

t0 = time.time()
parallel_test(10_000_000)
print(f"Parallel test done in {time.time()-t0:.3f}s")