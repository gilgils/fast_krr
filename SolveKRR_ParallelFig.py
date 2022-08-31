import figtree
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from multiprocessing import Pool
from scipy import linalg
from scipy.spatial.distance import cdist
import time


curr_precision = np.float64

def fig_helper(fig_params_dict):
    return figtree.figtree(**fig_params_dict)

# Divides N points into n baskets
def divide_batches_equally(N, n):
    div_result = N // n
    mod_result = N % n
    size = np.zeros((n + 1, 1)).astype(np.int32)
    size[1:] = div_result
    if mod_result > 0:
        size[1:mod_result + 1] += 1
    return np.cumsum(size)


# Applies a thin kernel to a matrix, divides the kernel by its rows (horizontally)
# For K(X,Y)q use parallel_fig_horiz(Y,X,q.T), typically Y.shape[0] > X.shape[0]
def parallel_fig_horiz(X, Y, Q, h=1, figtree_eps=1e-5, n_processes=8):
    size = divide_batches_equally(Y.shape[0], n_processes)
    parallel_list = [
        {'X': X, 'Y': Y[int(size[i]):int(size[i+1]), :], 'Q': Q.astype(curr_precision),
         'epsilon': figtree_eps, 'h': h} for i in range(n_processes)]
    with Pool(n_processes) as pool:
         res = pool.map(fig_helper, parallel_list)
    return np.hstack(res)

# split vertically
def parallel_fig_vert(X, Y, Q, h=1, figtree_eps=1e-5, n_processes=8):
    size = divide_batches_equally(Y.shape[0], n_processes)
    parallel_list = [{'X': X[int(size[i]):int(size[i+1]), :], 'Y': Y,
                      'Q': Q[:, int(size[i]):int(size[i+1])].astype(curr_precision), 'epsilon': figtree_eps,
                      'h': h} for i in range(n_processes)]
    with Pool(n_processes) as pool:
        res = np.sum(pool.map(fig_helper, parallel_list), axis=0)
    return res

def selectPoints_sparseID(X, l, k, h=1, figtree_eps=1e-5, n_processes=8):
    def generate_q():
        curr_q = np.random.randint(0, 2, 3)
        curr_q[curr_q == 0] = -1
        curr_q = curr_q[:, np.newaxis].astype(curr_precision)
        return curr_q
    parallel_list = [{'X': X[np.random.choice(X.shape[0], size=3, replace=False, p=None),:], 'Y': X, 'Q': generate_q(), 'epsilon': figtree_eps, 'h': h} for i in range(l)]
    with Pool(n_processes) as pool:
        res = pool.map(fig_helper, parallel_list)
    pool.join()
    pool.close()
    k_rand_proj = np.vstack(res)
    print("qr")
    _, _, p = linalg.qr(k_rand_proj, mode='economic', pivoting=True)
    return p[:k]

def build_Cholesky(X, anchors, noise, h, figtree_eps, n_processes):
    X_s = X[anchors,:]
    U = np.exp(-cdist(X_s, X_s, metric='seuclidean')/h**2)
    k = U.shape[0]
    u,s,v = np.linalg.svd(U)
    invUsqrt = np.matmul(u*(1/s**0.5), u.T)
    Y = parallel_fig_horiz(X=X_s, Y=X, Q=invUsqrt, h=h, figtree_eps=figtree_eps, n_processes=n_processes)
    Y = np.matmul(Y, Y.T) + noise*np.eye(k)
    L = linalg.cholesky(Y, lower=True)
    return L, invUsqrt
    
def ApplyPreconditioner(X, anchors, noise, L, invUsqrt, x, h, figtree_eps=1e-5, n_processes=8):
    X_s = X[anchors,:]
    y = invUsqrt.dot(parallel_fig_vert(X, X_s, x.T, h, figtree_eps, n_processes).T)
    z = invUsqrt.dot(linalg.cho_solve((L,True), y))
    return 1/noise*x.T - 1/noise*parallel_fig_horiz(X_s, X, z.T, h, figtree_eps, n_processes)


def ApplyOperator(X, noise, x, h, figtree_eps, n_processes):
    return parallel_fig_vert(X, X, x.T, h, figtree_eps, n_processes).T + noise*x

def ComputeModel(X, anchors, noise, x0, b, h, MaxIter=500, tol=1e-2, n_processes=8):
    L, invUsqrt = build_Cholesky(X, anchors, noise, h, figtree_eps, n_processes)
    x = x0
    r = b - ApplyOperator(X, noise, x, h, figtree_eps, n_processes)
    z = ApplyPreconditioner(X, anchors, noise, L, invUsqrt, r, h, figtree_eps, n_processes).T
    p = z
    k = 0
    err = np.inf
    all_x = list()
    all_err = list()
    while (err > tol) and (k < MaxIter):
        Ap = ApplyOperator(X, noise, p, h, figtree_eps, n_processes)
        alpha = np.sum(r*z)/np.sum(Ap*p)
        x_next = x + alpha*p
        r_next = r - alpha*Ap
        err = np.linalg.norm(r)
        print('err=', err, flush=True)
        z_next = ApplyPreconditioner(X, anchors, noise, L, invUsqrt, r_next, h, figtree_eps, n_processes).T
        beta = np.sum(z_next*(r_next-r))/np.sum(z*r)
        p_next = z_next + beta*p
        k += 1
        p = p_next
        r = r_next
        x = x_next
        z = z_next
        all_x.append(x)
        all_err.append(err)
    return x, all_x, all_err
        



if __name__=='__main__':
    #pool = mp.Pool(mp.cpu_count())
    N = 4000
    X = np.random.randn(N,3)
    x = np.zeros((N,1))
    b = np.random.randn(N, 1)
    t = time.time()
    l = 200
    k = l-5
    noise = 1
    h = 0.3
    n_processes = 8
    figtree_eps = 1e-12
    #z = parallel_fig_vert(X, X, x.copy(), h=1, figtree_eps=1e-5, n_processes=8)
    anchors = selectPoints_sparseID(X, l, k, h, figtree_eps, n_processes)
    x, all_x, all_err = ComputeModel(X, anchors, noise, x.copy(), b, h, MaxIter=500, tol=1, n_processes=8)

    K = np.exp(-cdist(X,X, metric='euclidean')**2 / h**2) + noise*np.eye(N)
    x_direct = np.linalg.pinv(K).dot(b)
    print(np.linalg.norm(x-x_direct))
  
