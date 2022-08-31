from FastGaussianKRR import *
import numpy as np
from scipy.spatial.distance import cdist
import time

d = 24
#N=18000
N=5000
X = np.genfromtxt('/home/gil/Downloads/sm_full_X.csv', delimiter=',')[0:N,:]
b = np.genfromtxt('/home/gil/Downloads/sm_full_X.csv', delimiter=',')[0:N,np.newaxis]
Y = np.random.randn(50, d)
v = np.random.randn(N, 1)
x = np.zeros((N, 1))
b = np.random.randn(N, 1)
t = time.time()
p = 5
k = 256
l = k + p
noise = 0.001
h = 1
n_processes = 8
figtree_eps = 1e-10
krr = FastGaussKRR(X=X, noise=noise, h=h, n_processes=8, curr_precision=np.float64, figtree_eps=figtree_eps,   tol=1e-4)
#t = krr.parallel_fig_vert(X, X[0:50], v)
krr.fit(k, 5, b)
pred = krr.predict(Y)
x = krr.solution
t0 = time.time()
K = np.exp(-cdist(X, X, metric='euclidean') ** 2 / h ** 2) + noise * np.eye(N)
x_direct = np.linalg.pinv(K).dot(b)
t_done = time.time()
print('Time for direct computation=', t_done-t0)
print(np.linalg.norm(x - x_direct) / np.linalg.norm(x_direct))
K_pred = np.exp(-cdist(Y, X, metric='euclidean') ** 2 / h ** 2)
pred_direct = K_pred.dot(x_direct)
print(linalg.norm(pred - pred_direct) / linalg.norm(pred_direct))
