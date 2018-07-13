	import time
	import numpy as np
	import cupy as cp
	from sys import getsizeof

	c1 = np.random.randn(10000, 10000)
	c2 = np.random.randn(10000,  10000)
	t1 = time.clock()
	o = c1.dot(c2)
	print(f"Numpy time : {round((time.clock()-t1), 4)} second ;o size = {round(getsizeof(o)/2**20, 4)} MB")
	c1 = cp.random.randn(10000, 10000)
	c2 = np.random.randn(10000, 10000)
	t2 = time.clock()
	o = c1.dot(c2)
	print(f"for loop time : {round((time.clock()-t2), 4)} second ;o size = {round(getsizeof(o)/2**20, 4)} MB")

