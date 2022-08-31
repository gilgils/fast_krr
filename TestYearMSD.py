from FastGaussianKRR import *
import numpy as np
from scipy.spatial.distance import cdist
import time
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
raw_data = np.genfromtxt('/home/gil/Downloads/YearPredictionMSD.txt', delimiter=',', max_rows=150000)
n = 5000
data = raw_data[0:n, 1:].astype(np.float64).copy()
b = raw_data[0:n, 0][:, np.newaxis].astype(np.float64).copy()

scaler.fit(data)
X = scaler.transform(data)

p = 500



k = 1000
l = k + p
noise = 0.001
h = 5
n_processes = 8
figtree_eps = 1e-10
krr = FastGaussKRR(X=X, noise=noise, h=h, n_processes=8, curr_precision=np.float64, figtree_eps=figtree_eps,   tol=1e-4)
#t = krr.parallel_fig_vert(X, X[0:50], v)
krr.fit(k, p, b)

