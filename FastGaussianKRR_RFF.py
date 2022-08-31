import figtree
import numpy as np
from multiprocessing import Pool
from scipy import linalg
from scipy.spatial.distance import cdist
from datetime import datetime
from random import sample


def fig_helper(fig_params_dict):
    return figtree.figtree(**fig_params_dict)


class FastGaussKRR_RFF:
    def __init__(self, X, noise, h, n_processes, curr_precision=np.float64, rff_scale=10, figtree_eps=1e-8, verbose=True, tol=0.01, max_iter=300, loopy_vert_divide_by=10):
        self.noise = noise
        self.X = X
        self.curr_precision = curr_precision
        self.figtree_eps = figtree_eps
        self.n_processes = n_processes
        self.h = h
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.scale = rff_scale
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

    def build_RFF_precond_matrix(self):
        d = np.shape(self.X)[1]
        D = self.scale*d
        W = np.random.randn(d, D) / (self.h / 2**0.5)
        Z = np.hstack((np.cos(self.X.dot(W)), np.sin(self.X.dot(W)))) / D**0.5 # Random Fourier features matrix
        L = np.linalg.cholesky(Z.T.dot(Z) + self.noise*np.eye(2*D))
        self.U = np.linalg.pinv(L).dot(Z.T)
        print('U=', self.U.shape)
        print('L=', L.shape)
        print('Z=', Z.shape)

    def apply_preconditioner(self, x):
        return (1/self.noise*(x - self.U.T.dot(self.U.dot(x)))).T

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
        self.print_log('Building RFF preconditioner...')
        start_precond = datetime.now()
        self.build_RFF_precond_matrix()
        done_precond = datetime.now()
        self.print_log('Done building preconditioner')
        x0 = np.zeros((self.X.shape[0], 1))
        self.print_log('Starting iterations')
        conjgrad_start = datetime.now()
        self.compute_model(x0.copy())
        conjgrad_end = datetime.now()
        self.print_log('Done iterations')
        self.rff_time = done_precond - start_precond
        self.conjgrad_time = conjgrad_end - conjgrad_start

    def predict(self, Y):
        return self.parallel_fig_vert(X=self.X, Y=Y, Q=self.solution.T).T