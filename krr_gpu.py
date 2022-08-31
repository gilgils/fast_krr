import torch
import numpy as np
from scipy.linalg import qr
import time

PRECISION = torch.float32
device = torch.device('cuda:0')

class block_krr:
    def __init__(self, X, noise, h, l, k, PRECISION = torch.float32, verbose=True, tol=0.01, max_iter=300, device=torch.device('cuda:0'), block_size=1000):
        self.X = X
        self.noise = noise
        self.h = h
        self.PRECISION = PRECISION
        self.verbose = verbose
        self.tol = tol
        self.l = l
        self.k = k
        self.max_iter = max_iter
        self.device = device
        self.block_size = block_size
        self.anchors = []
        self.invUsqrt = 0
        self.L = 0

    def build_kernel(self, X, Y, compute_mode='use_mm_for_euclid_dist_if_necessary'):
        # use_mm_for_euclid_dist_if_necessary, donot_use_mm_for_euclid_dist
        # compute_mode = 'donot_use_mm_for_euclid_dist'
        return torch.exp(-torch.cdist(X, Y, p=2, compute_mode=compute_mode) ** 2 / self.h ** 2)

    def blocked_fig_matrix(self, X, Y, q, compute_mode='use_mm_for_euclid_dist_if_necessary'):
        m,d = X.shape
        n,dummy = Y.shape
        dummy, l = q.shape
        out = torch.zeros((m, l)).to(device)
        num_iter = int(np.ceil(m/self.block_size))
        for i in range(num_iter):
            #print(i)
            start = i*self.block_size
            end = min(m, (i+1)*self.block_size)
            # subMatrix = torch.exp(-torch.cdist(X[start:end ,:], Y, p=2)**2 / self.h**2).to(self.device)
            subMatrix = self.build_kernel(X[start:end, :], Y, compute_mode=compute_mode).to(self.device)
            out[start:end,:] = torch.matmul(subMatrix, q)
        return out

    def select_points_randomized_ID(self):
        n,d = self.X.shape
        randMatrix = torch.randn((n,self.l), dtype=self.PRECISION).to(self.device)
        Projected = self.blocked_fig_matrix(self.X, self.X, randMatrix)
        # (_, _, p) = torch.qr(torch.transpose(Projected, 1, 0), pivot=True)
        _, _, p = qr(np.transpose(Projected.cpu().numpy()), pivoting=True)
        anchors = p[0:self.k]
        anchors.dtype = np.int32
        self.anchors = anchors

    def build_Cholesky(self):
            X_s = self.X[self.anchors,:]
            # U = torch.exp(-torch.cdist(X_s, X_s, p=2)**2/self.h**2).to(self.device)
            U = self.build_kernel(X_s, X_s, compute_mode='donot_use_mm_for_euclid_dist').to(self.device)
            k = U.shape[0]
            u,s,v = torch.svd(U) # matrix is symmetric
            self.invUsqrt = torch.matmul(u*(1/s**0.5), torch.transpose(u, 1, 0))
            Y = self.blocked_fig_matrix(self.X, X_s, self.invUsqrt).type(self.PRECISION)
            Y = torch.matmul(torch.transpose(Y, 1, 0), Y) + self.noise*torch.eye(k).to(self.device)
            # Y = torch.matmul(invUsqrt, blocked_fig_matrix(X_s, X, Y, BlockSize, h)) + noise*torch.eye(k)
            self.L = torch.linalg.cholesky(Y)

    def apply_preconditioner_operator(self, x):
        y = torch.matmul(self.invUsqrt, self.blocked_fig_matrix(self.X[self.anchors,:], self.X, x).type(self.PRECISION))
        z = torch.matmul(self.invUsqrt, torch.cholesky_solve(y, self.L))
        return 1.0/self.noise*x - 1.0/self.noise*self.blocked_fig_matrix(self.X, self.X[self.anchors,:], z)

    def apply_operator(self, x):
        return self.blocked_fig_matrix(self.X, self.X, x) + self.noise*x

    def compute_model(self, x0, b):
        if self.verbose:
            print('Anchor selection...')
        self.select_points_randomized_ID()
        if self.verbose:
            print('Building Cholesky')
        self.build_Cholesky()
        if self.verbose:
            print('PCG started...')
        x = x0
        r = b - self.apply_operator(x)
        z = self.apply_preconditioner_operator(r)
        p = z
        k = 0
        err = np.inf
        all_x = list()
        all_r = list()
        while (err > self.tol) and (k < self.max_iter):
            Ap = self.apply_operator(p)
            alpha = torch.sum(r*z)/torch.sum(p*Ap)
            x_next = x + alpha*p
            r_next = r - alpha*Ap
            err = torch.norm(r, p=2)
            if self.verbose:
                print('err=', err)
            z_next = self.apply_preconditioner_operator(r_next)
            beta = torch.sum(z_next*(r_next-r))/torch.sum(z*r)
            p_next = z_next + beta*p
            k += 1
            p = p_next
            r = r_next
            x = x_next
            z = z_next
            all_x.append(x)
            all_r.append(err)
        return x, all_x, all_r

    def predict(self, Y, x):
        return self.blocked_fig_matrix(Y, self.X, x)
