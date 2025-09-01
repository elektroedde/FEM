import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import warnings

# ---------------------------
# INPUT
# ---------------------------

nodes_xy = pd.read_csv("NODE_XY.csv", header=None).to_numpy(dtype=float)
N = nodes_xy.shape[0]
print("Total number of nodes:", N)

elements_index = pd.read_csv("ELEMENTS.csv", header=None).to_numpy(dtype=int)
elements_index = elements_index - 1  # convert to 0-based indexing
M = elements_index.shape[0]
print("Total number of elements:", M)

# material / source (can be scalars or arrays of length M later)
alpha_x = 1.0
alpha_y = 1.0
f_val = 1.0   # source term (constant for now)

# ---------------------------
# INITIALIZE GLOBAL SYSTEM
# ---------------------------

K = np.zeros((N, N), dtype=float)
b = np.zeros(N, dtype=float)

# ---------------------------
# ASSEMBLE ELEMENTS (linear triangles)
# ---------------------------

for e in range(M):
    nid = elements_index[e, :]           # three node indices (0-based)
    x = nodes_xy[nid, 0]
    y = nodes_xy[nid, 1]

    # geometry coefficients b_i, c_i (notation from standard linear triangle)
    b_geo = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c_geo = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

    area2 = b_geo[0]*c_geo[1] - b_geo[1]*c_geo[0]
    area = 0.5 * area2

    if abs(area) < 1e-12:
        warnings.warn(f"Element {e} has near-zero area ({area}). Skipping.")
        continue

    # Use absolute area for load; for stiffness the orientation sign cancels if used consistently.
    A = abs(area)

    # local stiffness matrix (Ke)
    # Ke_ij = (alpha_x * b_i b_j + alpha_y * c_i c_j) / (4 * area)
    Ke = (alpha_x * np.outer(b_geo, b_geo) + alpha_y * np.outer(c_geo, c_geo)) / (4.0 * area)

    # local load vector (fe) for a constant f over element: fe_i = f * area / 3
    fe = np.ones(3, dtype=float) * (f_val * A / 3.0)

    # assemble to global
    for a in range(3):
        A_glob = nid[a]
        b[A_glob] += fe[a]
        for c in range(3):
            C_glob = nid[c]
            K[A_glob, C_glob] += Ke[a, c]

# ---------------------------
# DIRICHLET BCs (cleanly)
# ---------------------------
def apply_dirichlet(K, b, node_index, value):
    """
    Enforce phi(node_index) = value on system K * phi = b.
    We first modify the RHS: b = b - K[:,node]*value,
    then zero the row/col, set K[node,node]=1 and b[node]=value.
    """
    # subtract column contribution from b
    b -= K[:, node_index] * value

    # zero row and column
    K[node_index, :] = 0.0
    K[:, node_index] = 0.0

    # set diag and rhs
    K[node_index, node_index] = 1.0
    b[node_index] = value


# Read DIRICHLET files if present and apply
# Format expected: col0 = global node number (1-based), col1 = prescribed phi
try:
    df = pd.read_csv("DIRICHLET1.csv", header=None)
    dir_nodes = df.to_numpy(dtype=float)
    dir_nodes[:, 0] = dir_nodes[:, 0].astype(int) - 1  # to 0-based
    for row in dir_nodes:
        ni = int(row[0])
        val = float(row[1])
        apply_dirichlet(K, b, ni, val)
    print("Applied DIRICHLET1:", dir_nodes.shape[0], "nodes")
except FileNotFoundError:
    print("DIRICHLET1.csv not found -- skipping")

try:
    df = pd.read_csv("DIRICHLET0.csv", header=None)
    dir_nodes = df.to_numpy(dtype=float)
    dir_nodes[:, 0] = dir_nodes[:, 0].astype(int) - 1  # to 0-based
    for row in dir_nodes:
        ni = int(row[0])
        val = float(row[1])
        apply_dirichlet(K, b, ni, val)
    print("Applied DIRICHLET0:", dir_nodes.shape[0], "nodes")
except FileNotFoundError:
    print("DIRICHLET0.csv not found -- skipping")

# ---------------------------
# SOLVE
# ---------------------------

# Optionally check conditioning
cond = np.linalg.cond(K)
print("Matrix condition number (approx):", cond)
if cond > 1e12:
    warnings.warn("Global stiffness matrix is ill-conditioned. Results may be inaccurate.")

phi = np.linalg.solve(K, b)

# ---------------------------
# PLOT (triangular mesh)
# ---------------------------

triang = tri.Triangulation(nodes_xy[:, 0], nodes_xy[:, 1], elements_index)

fig, ax = plt.subplots(figsize=(8, 8))
tcf = ax.tricontourf(triang, phi, cmap="jet", levels=12)
plt.colorbar(tcf, ax=ax, label="φ value")
ax.triplot(triang, color="k", alpha=0.25, linewidth=0.5)
ax.set_aspect("equal")
ax.set_title("Solution φ over mesh")
plt.tight_layout()
plt.show()
