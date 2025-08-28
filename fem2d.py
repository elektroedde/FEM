import numpy as np
def apply_dirichlet_bc(K, d, node, value):
    n = K.shape[0]

    # Zero out row
    K[node, :] = 0
    # Zero out column, but adjust RHS
    for i in range(n):
        d[i] -= K[i, node] * value
        K[i, node] = 0

    # Set diagonal and RHS
    K[node, node] = 1
    d[node] = value


def apply_third_kind_boundary(ns, K, d):

    segments = ns.shape[0]

    K_s = np.full((2, 2), 1/6) + np.eye(2) * (1/6)
    d_s = np.full(2, 1/2)

    for s in range(segments):
        for i in range(2):
            ni = ns[s, i] - 1 
            d[ni] += d_s[i]
            for j in range(2):
                nj = ns[s, j] - 1
                K[ni, nj] += K_s[i, j]
