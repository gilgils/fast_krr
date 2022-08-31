import torch
import numpy as np
from scipy.linalg import qr
from random import sample
import time

PRECISION = torch.float32
device = torch.device('cuda:0')

def blocked_fig_matrix(X, Y, q, BlockSize=1000, h=1):
    m,d = X.shape
    n,dummy = Y.shape
    dummy, l = q.shape
    out = torch.zeros((m, l)).to(device)
    num_iter = int(np.ceil(m/BlockSize))
    for i in range(num_iter):
        #print(i)
        start = i*BlockSize
        end = min(m, (i+1)*BlockSize)
        subMatrix = torch.exp(-torch.cdist(X[start:end ,:], Y, p=2)**2 / h**2).to(device)
        out[start:end,:] = torch.matmul(subMatrix, q)
    return out

def select_points_randomized_ID(X, l, k, BlockSize, h):
    n,d = X.shape
    randMatrix = torch.randn((n,l), dtype=PRECISION).to(device)
    Projected = blocked_fig_matrix(X, X, randMatrix, BlockSize, h) + noise*randMatrix
    #(_, _, p) = torch.qr(torch.transpose(Projected, 1, 0), pivot=True)
    _, _, p = qr(np.transpose(Projected.cpu().numpy()), pivoting=True)
    anchors = p[0:k]
    anchors.dtype = np.int32
    return anchors


def select_points_sparse_randomized_ID(X, l, k, BlockSize, h):
    r = 3
    n,d = X.shape
    Y = torch.zeros(n,l)
    for i in range(l):
        R = sample(range(n), r)
        v = torch.randn(r,1)
        v[v < 0] = -1
        v[v >= 0] = 1
        X_sub = self.


    randMatrix = torch.randn((n,l), dtype=PRECISION).to(device)
    Projected = blocked_fig_matrix(X, X, randMatrix, BlockSize, h) + noise*randMatrix
    #(_, _, p) = torch.qr(torch.transpose(Projected, 1, 0), pivot=True)
    _, _, p = qr(np.transpose(Projected.cpu().numpy()), pivoting=True)
    anchors = p[0:k]
    anchors.dtype = np.int32
    return anchors

def build_Cholesky(X, anchors, noise, h, BlockSize=1000):
        X_s = X[anchors,:]
        U = torch.exp(-torch.cdist(X_s, X_s, p=2)**2/h**2).to(device)
        k = U.shape[0]
        u,s,v = torch.svd(U) # matrix is symmetric
        invUsqrt = torch.matmul(u*(1/s**0.5), torch.transpose(u, 1, 0))
        Y = blocked_fig_matrix(X, X_s, invUsqrt, BlockSize, h)
        Y = torch.matmul(torch.transpose(Y, 1, 0), Y) + noise*torch.eye(k).to(device)
        #Y = torch.matmul(invUsqrt, blocked_fig_matrix(X_s, X, Y, BlockSize, h)) + noise*torch.eye(k)
        L = torch.cholesky(Y)
        return L, invUsqrt

def ApplyPreconditionerOperator(X, anchors, noise, L, invUsqrt, x, h, BlockSize=1000):
    y = torch.matmul(invUsqrt, blocked_fig_matrix(X[anchors,:], X, x, BlockSize, h))
    z = torch.matmul(invUsqrt, torch.cholesky_solve(y, L))
    return 1/noise*x - 1/noise*blocked_fig_matrix(X, X[anchors,:], z, BlockSize, h)

def ApplyOperator(X, noise, x, h, BlockSize=1000):
    return blocked_fig_matrix(X, X, x, BlockSize, h) + noise*x

def ComputeModel(X, anchors, noise, x0, b, h, BlockSize=1000, MaxIter=500, tol=1e-2):
    L, invUsqrt = build_Cholesky(X, anchors, noise, h, BlockSize)
    x = x0
    r = b - ApplyOperator(X=X, noise=noise, x=x, h=h, BlockSize=BlockSize)
    z = ApplyPreconditionerOperator(X=X, anchors=anchors, noise=noise, L=L, invUsqrt=invUsqrt, x=r, h=h, BlockSize=BlockSize)
    p = z
    k = 0
    err = np.inf
    all_x = list()
    all_r = list()
    while (err > tol) and (k < MaxIter):
        Ap = ApplyOperator(X=X, noise=noise, x=p, h=h, BlockSize=BlockSize)
        alpha = torch.sum(r*z)/torch.sum(p*Ap)
        x_next = x + alpha*p
        r_next = r - alpha*Ap
        err = torch.norm(r, p=2)
        print('err=', err)
        z_next = ApplyPreconditionerOperator(X=X, anchors=anchors, noise=noise, L=L, invUsqrt=invUsqrt, x=r_next, h=h, BlockSize=BlockSize)
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

def predict(X, Y, x, h, BlockSize=1000):
    return blocked_fig_matrix(Y, X, x, BlockSize, h)

n = 300000
d = 24
noise = 0.01
h = 1
k = 100
X = torch.randn((n, d), dtype=PRECISION).to(device)
b = torch.randn((n, 1), dtype=PRECISION).to(device)
anchors = select_points_randomized_ID(X, l=k+5, k=k, BlockSize=1000, h=h)
x0 = torch.zeros((n, 1)).to(device)
x, x_all, r_all = ComputeModel(X=X, anchors=anchors, noise=noise, x0=x0.clone(), b=b, h=h, BlockSize=1000, MaxIter=200, tol=1e-2)
print(x)

K = torch.exp(-torch.cdist(X, X, p=2)**2 / h**2) + noise*torch.eye(n).to(device)
Kinv = torch.inverse(K)
x_direct = torch.matmul(Kinv, b)
print(x_direct)
print('done')


