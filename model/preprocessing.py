import time
import numpy as np
import cupy as cp
from sys import getsizeof
t1 = time.clock()

for i in range(500):

	c1 = np.random.randn(1000, 1000)
	c2 = np.random.randn(1000,  1000)
	o = c1.dot(c2)

print(f"Numpy time : {round((time.clock()-t1), 4)} second ;o size = {round(getsizeof(o)/2**20, 4)} MB")
t2 = time.clock()

for i in range(500):
	c1 = cp.random.randn(1000, 1000)
	c2 = cp.random.randn(1000, 1000)
	o = c1.dot(c2)



print(f"for loop time : {round((time.clock()-t2), 4)} second ;o size = {round(getsizeof(o)/2**20, 4)} MB")

