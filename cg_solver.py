import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt

# Define the source function
def f_func(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Build the sparse matrix A for the 2D Poisson problem (5-point stencil)
def build_poisson_matrix(N):
    n = N - 1
    e = np.ones(n)
    T = sp.diags([e, -2 * e, e], offsets=[-1, 0, 1], shape=(n, n))
    I = sp.eye(n)
    A = sp.kron(I, T) + sp.kron(T, I)
    return -A

# Build the right-hand side vector b
def build_rhs(N, f):
    n = N - 1
    h = 1.0 / N
    b = np.zeros(n * n)
    for j in range(1, N):
        y = j * h
        for i in range(1, N):
            x = i * h
            idx = (j - 1) * n + (i - 1)
            b[idx] = h**2 * f(x, y)
    return b

# Conjugate gradient solver with random initial guess
def cg_solver(A, b, tol=1e-8, max_iter=10000, perturbation=1e-3):
    x = perturbation * np.random.randn(b.shape[0])
    r = b - A.dot(x)
    p = r.copy()
    rsold = r.dot(r)
    rs0 = rsold

    for it in range(max_iter):
        Ap = A.dot(p)
        alpha = rsold / p.dot(Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, it + 1, np.sqrt(rs0), np.sqrt(rsnew)

# Convert 1D solution vector back to a 2D grid 
def reconstruct_solution(x, N):
    n = N - 1
    u = np.zeros((N + 1, N + 1))
    u[1:N, 1:N] = x.reshape((n, n))
    return u

if __name__ == "__main__":
    grid_sizes = [8, 16, 32, 64, 128, 256]
    summary = []

    for N in grid_sizes:
        print(f"\nSolving for N = {N}")
        A = build_poisson_matrix(N)
        b = build_rhs(N, f_func)

        start = time.time()
        x, iterations, initial_res, final_res = cg_solver(
            A, b, tol=1e-8, max_iter=10000, perturbation=1e-3
        )
        elapsed = time.time() - start

        print(f"Initial residual norm = {initial_res:.2e}, "
              f"Final residual norm = {final_res:.2e}, "
              f"Iterations = {iterations}, Time = {elapsed:.4f}s")
        summary.append((N, iterations, elapsed))

        # Plot the computed solution 
        u = reconstruct_solution(x, N)
        xv = np.linspace(0, 1, N + 1)
        yv = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(xv, yv)

        plt.figure()
        cf = plt.contourf(X, Y, u, levels=20, cmap='viridis')
        plt.colorbar(cf, label='u(x,y)')
        plt.title(f"Solution for N = {N}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.savefig(f"solution_N{N}.png", dpi=300)
