import figtree
import numpy as np
from multiprocessing import Pool
from scipy import linalg
from scipy.spatial.distance import cdist
from datetime import datetime
from random import sample


def fig_helper(fig_params_dict):
    return figtree.figtree(**fig_params_dict)


class FastGaussKRR:
    def __init__(self, X, noise, h, n_processes, curr_precision=np.float32, figtree_eps=1e-8, verbose=True, tol=0.01, max_iter=300, loopy_vert_divide_by=10):
        self.noise = noise
        self.X = X
        self.curr_precision = curr_precision
        self.figtree_eps = figtree_eps
        self.n_processes = n_processes
        self.h = h
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.anchors = None
        self.loopy_vert_divide_by = loopy_vert_divide_by

    def print_log(self, message):
        if self.verbose:
            print("{} - ".format(datetime.now()), message, flush=True)


    # Divides N points into n baskets

    @staticmethod
    def divide_batches_equally(num_examples, num_batches):
        div_result = num_examples // num_batches
        mod_result = num_examples % num_batches
        size = np.zeros((num_batches + 1, 1)).astype(np.int32)
        size[1:] = div_result
        if mod_result > 0:
            size[1:mod_result + 1] += 1
        return np.cumsum(size)


    # Applies a thin kernel to a matrix, divides the kernel by its rows (horizontally)
    # For K(X,Y)q use parallel_fig_horiz(Y,X,q.T), typically Y.shape[0] > X.shape[0]

    def parallel_fig_horiz(self, X, Y, Q):
        if Y.shape[0] < self.n_processes:
            n_active = Y.shape[0]
        else:
            n_active = self.n_processes
        size = self.divide_batches_equally(Y.shape[0], n_active)
        parallel_list = [
            {'X': X, 'Y': Y[int(size[i]):int(size[i+1]), :], 'Q': Q.astype(self.curr_precision).copy(),
             'epsilon': self.figtree_eps, 'h': self.h} for i in range(n_active)]
        with Pool(n_active) as pool:
            res = pool.map(fig_helper, parallel_list)
        return np.hstack(res)

    # split vertically
    def parallel_fig_vert(self, X, Y, Q):
        if X.shape[0] < self.n_processes:
            n_active = X.shape[0]
        else:
            n_active = self.n_processes
        size = self.divide_batches_equally(X.shape[0], n_active)
        parallel_list = [{'X': X[int(size[i]):int(size[i+1]), :], 'Y': Y,
                          'Q': Q[:, int(size[i]):int(size[i+1])].astype(self.curr_precision).copy(), 'epsilon': self.figtree_eps,
                          'h': self.h} for i in range(n_active)]
        with Pool(n_active) as pool:
            res = np.sum(pool.map(fig_helper, parallel_list), axis=0)
        return res

    def loopy_parallel_fig_vert(self, X, Y, Q):
        blocks = self.divide_batches_equally(Q.shape[0], self.loopy_vert_divide_by)
        out = np.zeros((Y.shape[0], Q.shape[1]))
        for j in range(self.loopy_vert_divide_by):
            out[:, blocks[j]:blocks[j + 1]] = self.parallel_fig_vert(X, Y, Q[:, blocks[j]:blocks[j + 1]].T).T
        return out.T

    def generate_q(self):
        curr_q = np.random.randint(0, 2, 3)
        curr_q[curr_q == 0] = -1
        curr_q = curr_q[:, np.newaxis].astype(self.curr_precision)
        return curr_q

    def select_points_sparse_id(self, l, k):
        parallel_list = [{'X': self.X[np.random.choice(self.X.shape[0], size=3, replace=False, p=None),:], 'Y': self.X, 'Q': self.generate_q(), 'epsilon': self.figtree_eps, 'h': self.h} for i in range(l)]
        with Pool(self.n_processes) as pool:
            res = pool.map(fig_helper, parallel_list)
        k_rand_proj = np.vstack(res)
        _, _, p = linalg.qr(k_rand_proj, mode='economic', pivoting=True)
        self.anchors = p[:k]

    def select_points_sparse_parallel_vert(self, l, k, s=3):
        n = self.X.shape[0]
        out = np.zeros((n, k))
        for i in range(k):
            #print(i)
            chosen_s = sample(range(n), s)
            rand_vec = np.sign(np.random.randn(s, 1))
            v = self.parallel_fig_vert(self.X[chosen_s, :], self.X, rand_vec.T).T
            # v = self.parallel_fig_horiz(self.X[chosen_s, :], self.X, rand_vec.T).T
            out[:, i][:, np.newaxis] = v
        _, _, p = linalg.qr(out, mode='economic', pivoting=True)
        self.anchors = p[:k]

    def build_cholesky(self):
        X_s = self.X[self.anchors,:]
        U = np.exp(-cdist(X_s, X_s, metric='sqeuclidean')/self.h**2)
        k = U.shape[0]
        self.print_log('Performing SVD on U...')
        u, s, v = np.linalg.svd(U)
        self.print_log('Finished SVD on U')
        self.invUsqrt = np.matmul(u*(1/s**0.5), u.T) # make sure s^(1/2) doesn't divide by zero
        self.print_log('Calculating U^(-1/2)*C...')
        #invUsqrt_C = self.parallel_fig_horiz(X=X_s, Y=self.X, Q=self.invUsqrt).astype(np.float64)
        #invUsqrt_C = self.parallel_fig_vert(X=X_s, Y=self.X, Q=self.invUsqrt).astype(np.float64)
        invUsqrt_C = self.loopy_parallel_fig_vert(X=X_s, Y=self.X, Q=self.invUsqrt).astype(np.float64)
        self.print_log('Finished calculating U^(-1/2)*C')
        Y = np.matmul(invUsqrt_C, invUsqrt_C.T) + self.noise*np.eye(k)
        self.print_log('Calculating L for cholesky decomposition')
        self.L = linalg.cholesky(Y, lower=True).astype(self.curr_precision)

    def apply_preconditioner(self, x):
        X_s = self.X[self.anchors, :]
        y = self.invUsqrt.dot(self.parallel_fig_vert(self.X, X_s, x.T).T)
        z = self.invUsqrt.dot(linalg.cho_solve((self.L, True), y))
        return 1/self.noise*x.T - 1/self.noise*self.parallel_fig_vert(X_s, self.X, z.T)

    def apply_operator(self, x):
        return self.parallel_fig_vert(self.X, self.X, x.T).T + self.noise*x

    def compute_model(self, x_0):
        self.print_log('Tolerance: {}'.format(self.tol))
        x = x_0
        r = self.b - self.apply_operator(x)
        # print(r.shape)
        z = self.apply_preconditioner(r).T
        # print(z.shape)
        p = z
        k = 0
        err = np.inf
        all_x = list()
        all_err = list()
        Ax_list = []
        while (err > self.tol) and (k < self.max_iter):
            Ap = self.apply_operator(p)
            alpha = np.sum(r*z)/np.sum(Ap*p)
            x_next = x + alpha*p
            r_next = r - alpha*Ap
            if Ax_list == []:
                Ax_list.append(alpha * Ap)
            else:
                Ax_list.append(Ax_list[-1] + alpha * Ap)
            err = np.linalg.norm(r) / (len(r)**0.5)
            if self.verbose:
                self.print_log('RMSE at iteration {} = {}'.format(k, err))
            z_next = self.apply_preconditioner(r_next).T
            beta = np.sum(r_next*(z_next-z))/np.sum(z*r)
            p_next = z_next + beta*p
            k += 1
            p = p_next
            r = r_next
            x = x_next
            z = z_next
            all_x.append(x)
            all_err.append(err)
        np_ax_list = np.array(Ax_list[:-1])
        np_ax_list_subtracted = np_ax_list - Ax_list[-1]
        np_x_list = np.array(all_x[:-1])
        np_x_list_subtracted = np_x_list - all_x[-1]
        norm_array = np.multiply(np_ax_list_subtracted, np_x_list_subtracted).reshape(-1,np_ax_list_subtracted.shape[1]).sum(
            axis=1)
        self.solution = x
        self.all_x = all_x
        self.all_err = all_err
        self.norm_err = norm_array

    def fit(self, number_of_anchors, over_sampling, b):
        self.print_log('Starting (fit)...')
        self.b = b
        anchor_selection_start = datetime.now()
        self.print_log('Starting anchor selection')
        self.select_points_sparse_id(number_of_anchors + over_sampling, number_of_anchors)
        # self.anchors = np.arange(500)
        #self.select_points_sparse_parallel_vert(l=number_of_anchors + over_sampling, k=number_of_anchors)
        self.print_log('Done anchor selection')
        anchor_selection_end = datetime.now()
        self.print_log('Starting Cholesky')
        cholesky_start = datetime.now()
        self.build_cholesky()
        cholesky_end = datetime.now()
        self.print_log('Done Cholesky')
        x0 = np.zeros((self.X.shape[0], 1))
        self.print_log('Starting iterations')
        conjgrad_start = datetime.now()
        self.compute_model(x0.copy())
        conjgrad_end = datetime.now()
        self.print_log('Done iterations')
        self.anchor_selection_time = anchor_selection_end - anchor_selection_start
        self.cholesky_time = cholesky_end - cholesky_start
        self.conjgrad_time = conjgrad_end - conjgrad_start

    def predict(self, Y):
        return self.parallel_fig_vert(X=self.X, Y=Y, Q=self.solution.T).T