import numpy as np
import matplotlib.pyplot as plt

# Build dense matrix
def build_dense_matrix(N):
    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            A[i, j] = (N - abs(i - j)) / N
    return A

# CG solver
def cg_solver(A, b, reltol, max_iter=10000):
    x = np.zeros_like(b)
    r = b - A.dot(x)
    p = r.copy()
    r0_norm = np.linalg.norm(r)
    res = [r0_norm]
    rsold = r0_norm**2
    for k in range(1, max_iter+1):
        Ap = A.dot(p)
        alpha = rsold / (p.dot(Ap))
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.dot(r)
        r_norm = np.sqrt(rsnew)
        res.append(r_norm)
        if r_norm <= reltol * r0_norm:
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    return x, res

# Theoretical bound
def theoretical_bound(k, kappa):
    factor = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
    return 2 * factor**k

if __name__ == "__main__":
    Ns = [10**2, 10**3, 10**4, 10**5]
    for N in Ns:
        print(f"\nN = {N} ")
        try:
            A = build_dense_matrix(N)
        except MemoryError:
            print(f"Skip N={N}: not enough memory.")
            continue
        b = np.ones(N, dtype=np.float64)
        kappa = np.linalg.cond(A)
        print(f"Condition number : {kappa:.2e}")
        reltol = np.sqrt(np.finfo(float).eps)
        x, residuals = cg_solver(A, b, reltol)
        iters = len(residuals) - 1
        print(f"Iterations : {iters}")
        print(f"Final residual: {residuals[-1]:.2e}")
        print(f"Residual tolerance : {reltol * residuals[0]:.2e}")
        # Plot
        k_vals = np.arange(iters + 1)
        ratio = np.array(residuals) / residuals[0]
        bound = [theoretical_bound(k, kappa) for k in k_vals]
        plt.figure()
        plt.semilogy(k_vals, ratio, label="Actual normalized residual")
        plt.semilogy(k_vals, bound, '--', label="Theoretical bound")
        plt.xlabel("Iteration")
        plt.ylabel("Normalized residual")
        plt.title(f"CG Convergence (N={N})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"cg_convergence_N{N}.png", dpi=300)
        plt.show()
