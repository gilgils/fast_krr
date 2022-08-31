# Fast Kernel Ridge Regression using Matrix Factorizations for Preconditioning 
The code implements a fast preconditioned kernel ridge regression that can be applied to large datasets without memory limitations [[1]](#1).
There is a CPU version based on the fast improved Gauss transform (FIG), that has to be downloaded and compiled from the FIG repository. We have modified it slightly to support single and double precision, but for us it used as a third party package and is not being maintaned. This version is highly efficient for low dimensions, but it also works reasonable well on multicore CPUs.
In addition, there is a code that does not use the FIG-transform, but divides the kernel into blocks. This code can run on CPU/GPU and can be easily modified to work on any other kernel. That the code uses Pytorch's cdist for computing the distance matrix (torch.cdist), which allows matrix-vector multiplications. In some cases might produce results which are not very accurate. This option can be changed inside the code, but it will run slower.


## References
<a id="1">[1]</a> 
G. Shabat, E. Choshen, D. Ben Or, N. Carmel;
Fast and accurate Gaussian kernel ridge regression using matrix decompositions for preconditioning
SIAM Journal on Matrix Analysis and Applications, 42(3), pp. 1073--1095, 2021.
