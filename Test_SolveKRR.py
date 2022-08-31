from FastGaussianKRR import *
import numpy as np
from scipy.spatial.distance import cdist
import time

if __name__ == '__main__':
    # pool = mp.Pool(mp.cpu_count())
    d = 24
    N = 30000
    X = np.random.randn(N, d)
    Y = np.random.randn(50, d)
    x = np.zeros((N, 1))
    b = np.random.randn(N, 1)
    t = time.time()
    p = 5
    k = 500
    l = k + p
    noise = 0.001
    h = 1
    n_processes = 8
    figtree_eps = 1e-12
    krr = FastGaussKRR(X=X, noise=noise, h=h, n_processes=8, curr_precision=np.float64, figtree_eps=figtree_eps,
                       tol=1e-4)
    krr.fit(k, 5, b)
    pred = krr.predict(Y)
    x = krr.solution
    K = np.exp(-cdist(X, X, metric='euclidean') ** 2 / h ** 2) + noise * np.eye(N)
    x_direct = np.linalg.pinv(K).dot(b)
    print(np.linalg.norm(x - x_direct) / np.linalg.norm(x_direct))
    K_pred = np.exp(-cdist(Y, X, metric='euclidean') ** 2 / h ** 2)
    pred_direct = K_pred.dot(x_direct)
    print(linalg.norm(pred - pred_direct) / linalg.norm(pred_direct))
