from krr_gpu import *

n = 40000
d = 24
noise = 0.001
h = 1
k = 100
PRECISION = torch.float64
X = torch.randn((n, d), dtype=PRECISION).to(device)
b = torch.randn((n, 1), dtype=PRECISION).to(device)
krr = block_krr(X=X, noise=noise, h=h, PRECISION=PRECISION, verbose=True, l=12, k=10, tol=0.01, max_iter=300)
x0 = torch.zeros((n, 1), dtype=PRECISION).to(device)
x, x_all, r_all = krr.compute_model(x0=x0.clone(), b=b)
print(x)
