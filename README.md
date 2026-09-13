# High-Performance Computing Coursework
> Overall mark: 86/100

Contains two assessed Jupyter notebooks, containing some of the coursework I did for my Techniques for High-Performance Computing module I took in university.

The first implements a custom CSR matrix class with relevant benchmarking. The second solves the 2D heat equation on CPU and then on GPU with a custom CUDA kernel (and associated benchmarking).

> **Note on code quality:** These are coursework notebooks, written for Google Colab. They were completed under a deadline and tidied only lightly for presentation. I have written down notes on where I have chosen to stop optimising, as I have cleared the requested criteria, and optimising for the sake of optimising doesn’t make much sense to do in Python. If I want top efficiency, I would have to switch a lower-level language, or at least written my GPU code in full C++ vernacular via Numba CUDA.

Unlike a research repo, this one is self-contained, without external data, so both notebooks run E2E. I have written through Numba CUDA, and so an NVidia GPU is necessary to run this code (or use Google Colab [free tier suffices]).

## 01 - Sparse matrix formats
> Mark: 97/100
> Minor mistake in forgetting to use Numba NJIT and parallelisation in CSR implementation, and forgetting to explicitly state that the matrix used in benchmarking was symmetric positive-definite.

### Part I: A CSR matrix class built on SciPy
Compressed Sparse Row (CSR) matrix class used for optimal memory storage of large, sparse matrices. To discuss the benefits of the CSR format, we consider an $`N\times M`$ matrix, which has $`NNZ`$ number of non-zero entries. CSR memory efficiency runs $`\mathcal O(NNZ + M)`$, which is vastly superior to storing every matrix entry, which instead runs $`\mathcal O(N\times M)`$. Row access, and matrix-vector multiplication is far superior for sparse matrices too - they run at $`\mathcal O (NNZ)`$ and $`\mathcal O(NNZ)`$ respectively. If we require fast column access instead, we can define CSC in an analogous way, as CSR runs column access at $`\mathcal O(NNZ + M)`$ or worse. The real downfall of this format is element lookup and inserting new NNZ elements, which run $`\mathcal O\left(\log(NNZ)\right)`$ (vs. $`\mathcal O(1)`$ as dense format) and $`\mathcal O(NNZ)`$ (vs. $`\mathcal O(1)`$ as dense format) respectively.

Coordinate (COO) matrix class is far better at building matrices, and much easier to read. To build a COO matrix, we simply need arrays `row`, `col`, `data`, and every data point has its coordinates recorded in the `row`, `col` pair, with the value of the point recorded in `data`. For instance, a matrix with `(1, 2)` entry `(3.1415)`, will be recorded by appending `0` to `row`, appending `1` to `col`, and appending `3.1415` to `data`. For this reason, entry lookup and insertion both go back down to $`\mathcal O (1)`$, just like the dense format, but the COO format is worse for very sparse matrices, as the memory footprint is $`(3NNZ)`$, compared to CSR’s $`(2NNZ + M + 1)`$.

The custom CSR class takes a COO matrix, converts it to CSR and stores it. We then benchmark against usual `@` NumPy matrix-vector multiplication and see that despite the higher overhead on our function, it scales linearly and looks to perform better on matrices sized $`10^4`$ or larger. The subclass is built under `scipy.sparse.linalg.LinearOperator`, which means that SciPy accepts the custom implementation, and with correctly written `__add__` and `_matvec` operators written, can correctly run with any SciPy solvers. The notebooks successfully run SciPy `gmres` and `cg` solvers.

### Part II: Custom matrix class
The second class in this notebook stores a matrix of the form
```math
\begin{pmatrix}
\mathbf{diag}(D) & 0 \\
0 & TW
\end{pmatrix}
```
where $`T`$ is $`(n, 2)`$ and $`W`$ is $`(2, n)`$. Without ever forming the $`(n, n)`$ product $`TW`$, but only storing the two matrices together, we reduce an $`\mathcal O(n^2)`$ operation down to two $`\mathcal O(n)`$ operations. This custom class massively outperforms the standard SciPy COO method, which is only natural - no matter how optimised SciPy/NumPy matrix-vector multiplication is, it will eventually be outperformed by our new class due to the order-of-magnitude complexity reduction.

### Known limitation(s):
* The CSR construction uses `bincount`, which <ins>silently</ins> accepts (and sums) duplicate entries instead of rejecting them, or even flagging them. Well formed input is assumed, rather than validated.

## 02 - 2D heat equation, CPU and GPU
> Mark: 77/100
> The report was overall rushed and several benchmarking mistakes were made - most egregious of all was that there was no direct CPU vs. GPU comparison, and that my CPU section did not include any convergence testing. The implementations were correct and well-written - the only minor correction I received was once again forgetting to use Numba parallelisation for my CPU implementation (overall lost 4 marks due to this).

We solve the homogenous heat equation with Dirichlet boundary conditions on a rectangular boundary, and we are asked to find the time, $`t^*`$ when the centre of the plate reaches exactly $`1`$ a.u. (or degrees). The is a simple parabolic PDE, where one end of the plate is held at $`5`$ a.u., while the other three are held at $`0`$. The standard procedure is to use the usual Laplace 5-point stencil to resolve the boundaries encroaching on the spatial domain, and update the temporal domain using a choice of integrator.

### Part I: CPU
We chose the 5-point Laplacian stencil, and benchmarked 3 different time-integrators; Forward Euler (FEuler), Backward Euler (BEuler) [through a pre-computed LU factorisation], and 4th order Runge-Kutta (RK4). The first bit of optimisation we can immediately spot is a symmetric reduction of our spatial domain. Because we have two opposing sides of our domain bearing the same boundary condition, we can simply fold the plane into half across that centre-line, and place a Neumann boundary condition there instead. We use a ghost-point correction to our 5-point Laplace stencil, and this allows us to halve the number of computations we require. We follow up with some benchmarking and stability analysis on each time integrator.

### Part II: GPU
We only use a forward Euler step in this section; that’s because in the previous section we didn’t find much improvement over the other methods compared to how long they took to calculate. The obvious thing to optimise immediately is to set up a shared memory halo, since every entry will be accessed more than once (baring the border entries), this is a substantial speed-up. The rest of the optimisation I utilised lies in a carefully-written kernel - it’s important to carefully handle CPU-GPU communication because it is slow. The CPU (host) hands out instructions and needs to decide when the simulation must stop, and so we copy only a “single” value (pardon the pun - it was actually a double) to the CPU once the GPU computes it. This could have been sped up still by keeping a count of the number of steps taken and only copying over this value every, say 100,000 steps. Or we could have had a kernel/device function which handles the calculation on-device (on the GPU), and only returning a Boolean True/False, which is just 8 bits, rather than 8 Bytes, etc. We can continue to discuss other methods of optimising this programme, but as is true with optimisation in general - it is usually useless if not done without purpose, and the state of my code was good enough for submission (the report itself - not so much, as discussed above).

*Quite an interesting optimisation worth discussing is using a 9-point (Mehrstellen) stencil, and indeed I wrote a 9-point stencil kernel, but it seems I’ve misplaced it and it isn’t actually in this notebook? Either way, I expect that such a stencil will give a decent speed-up, which would allow the running of finer `dt`, and thus improve the accuracy by a good margin. Indeed the latter part of the report was correct in stating that we got lucky in our $`N=1001`$ case and got a low error, but incorrect in its reasoning why. At that scale, what we see was simple noise fluctuation due to a relatively large `dt`.*
