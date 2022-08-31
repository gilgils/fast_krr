from FastGaussianKRR import *
import numpy as np
from scipy.spatial.distance import cdist
import time
from sklearn.preprocessing import StandardScaler
from numba import jit

def loopy_parallel_fig_vert(X, Y, Q, divide_by):
    blocks = krr.divide_batches_equally(Q.shape[0], divide_by)
    out = np.zeros((Y.shape[0], Q.shape[1]))
    for j in range(divide_by):
        out[:, blocks[j]:blocks[j+1]] = krr.parallel_fig_vert(X, Y, Q[:, blocks[j]:blocks[j+1] ].T).T
    return out.T

scaler = StandardScaler()
m = 2000
n = 3000
d = 3

X = np.random.randn(m, d)
Y = np.random.randn(n, d)
Q = np.random.randn(m, m)

figtree_eps = 1e-12
h = 1
krr = FastGaussKRR(X=X, noise=1e-2, h=h, n_processes=8, curr_precision=np.float64, figtree_eps=figtree_eps,   tol=1e-4)

t = time.time()
z = krr.parallel_fig_vert(X, Y, Q.T).T
print('parallel_fig_vert time=',time.time() - t)

K = np.exp(-cdist(Y, X, 'sqeuclidean') / h**2)
zz = K.dot(Q)
#print(np.linalg.norm(z-zz))
#print('z size=', np.shape(z))
'''
out = np.zeros((1300, 5000))
t = time.time()
for j in range(5):
    print(j, flush=True)
    out[:, j*1000:(j+1)*1000] = krr.parallel_fig_vert(X, Y, x[:,j*1000:(j+1)*1000].T).T
tt = time.time()
print('iterative time=', tt-t)
print('error=', np.linalg.norm(out-z))
'''
t = time.time()
out =  loopy_parallel_fig_vert(X, Y, Q, 10).T
t2 = time.time()
print('loopy time=', t2-t)
print(np.linalg.norm(z-out))
print(np.linalg.norm(zz-out))
print(np.linalg.norm(zz-z))



                                   
#krr.fit(k, p, b)

