#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

import multiprocess as mp
import numpy as np
import time

np.random.seed(0)
A = np.random.rand(800,800)
B = np.random.rand(800,800)

def wrapper(i):
    return np.linalg.norm(A.dot(B))

t0 = time.perf_counter()
with mp.Pool(mp.cpu_count()) as pool:
    pool.map(wrapper, range(64))
print(time.perf_counter()-t0)
