# Fast Kernel Ridge Regression using Matrix Factorizations for Preconditioning.
The code implements a fast kernel ridge regression that can be applied to large datasets without memory limitations.
There is a CPU version based on the fast improved Gauss transform (FIG), that has to be downloaded and compiled from the FIG repository. We have modified it slightly to support single and double precision, but for us it is a third party package and not very maintained. This version is highly efficient for low dimensions, but it also works reasonable well on multicore CPUs.
In addition, there is a code that does not the FIG-transform, but divides the kernel into blocks. This code can run on CPU/GPU.
Note, that the code uses the matrix-multiplication for computing the distances using (torch.cdist), which in some cases might produce results which are not very accurate. This option can be changed inside the code, but it will run slower.

The code is based on the following paper:
G. Shabat, E. Choshen, D. Ben Or, N. Carmel; "Fast and accurate Gaussian kernel ridge regression using matrix decompositions for preconditioning" SIAM Journal on Matrix Analysis and Applications, 42(3), pp. 1073--1095, 2021.
