# Fast Kernel Ridge Regression using Matrix Factorizations for Preconditioning.
The code implements a fast kernel ridge regression that can be applied to large datasets without memory limitations.
There is a CPU version based on the fast improved Gauss transform (FIG), that has to be downloaded and compiled from the FIG repository. We have modified it slightly to support single and double precision, but for us it is a third party package and not very maintained. This version is highly efficient for low dimensions, but it also works reasonable well on multicore CPUs.
In addition, there is a code that does not the FIG-transform, but divides the kernel into blocks. This code can run on CPU/GPU.
Note, that the code uses the matrix-multiplication for computing the distances using (torch.cdist), which in some cases might produce results which are not very accurate. This option can be changed inside the code, but it will run slower.


@article{shabat2021fast,\\
  title={Fast and accurate Gaussian kernel ridge regression using matrix decompositions for preconditioning},\\
  author={Shabat, Gil and Choshen, Era and Or, Dvir Ben and Carmel, Nadav},\\
  journal={SIAM Journal on Matrix Analysis and Applications},\\
  volume={42},\\
  number={3},\\
  pages={1073--1095},\\
  year={2021},\\
  publisher={SIAM}\\
}
