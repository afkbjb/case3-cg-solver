import numpy as np
import scipy.sparse as sp

# Source function
def f_func(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Build sparse matrix A using 5-point stencil
def build_poisson_matrix(N):
    n = N - 1
    e = np.ones(n)
    T = sp.diags([e, -2*e, e], [-1, 0, 1], shape=(n, n))
    I = sp.eye(n)
    A = sp.kron(I, T) + sp.kron(T, I)
    return -A

# Build right-hand side vector b
def build_rhs(N, f):
    n = N - 1
    h = 1.0 / N
    b = np.zeros(n * n)
    for j in range(1, N):
        y = j * h
        for i in range(1, N):
            x = i * h
            k = (j - 1) * n + (i - 1)
            b[k] = h**2 * f(x, y)
    return b

def reconstruct_solution(x, N):
    n = N - 1
    u = np.zeros((N + 1, N + 1))
    u[1:N, 1:N] = x.reshape((n, n))
    return u
