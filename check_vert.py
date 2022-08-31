from FastGaussianKRR import *
import numpy as np
from scipy.spatial.distance import cdist
import time
import figtree

d = 3
n = 1024
m = n//2
A = np.random.randn(n, d).astype(np.float64)
B = np.random.randn(m, d).astype(np.float64)

krr = FastGaussKRR(X=A, noise=0.001, h=1, n_processes=8, curr_precision=np.float64, figtree_eps=1e-12,   tol=1e-4)

K = np.exp(-cdist(B, A, 'sqeuclidean'))
v = np.random.randn(n, 1)
r = K.dot(v)
#w = krr.parallel_fig_vert(B,A, v)
ww = krr.parallel_fig_dummy(A,B, v)


print(w)