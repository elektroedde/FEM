import numpy as np
import matplotlib.pyplot as plt

# === Problem setup ===
freq = 1e9
wavelength = 3e8/freq
k0 = 2*np.pi/wavelength
E0 = 1.0
theta_deg = 45.0
theta = np.deg2rad(theta_deg)

# Material parameters
er = 4.0 - 1j*0.1
ur = 2.0

# Slab thickness
L = 5*wavelength

# FEM mesh
M = 30000         # number of elements
N = M + 1        # number of nodes
le = L / M
nodes = np.linspace(0.0, L, N)

# === Element coefficients (uniform mesh, same per element) ===
alpha = [1.0/ur] * M
beta  = [-k0**2*(er - (np.sin(theta)**2)/ur)] * M
f     = [0.0] * M
l     = [le] * M

# === Assemble stiffness matrix and RHS ===
K = np.zeros((N, N), dtype=complex)
b = np.zeros(N, dtype=complex)

# Interior contributions
K[0,0] = alpha[0]/l[0] + beta[0]*l[0]/3
K[N-1,N-1] = alpha[M-1]/l[M-1] + beta[M-1]*l[M-1]/3

for i in range(1, N-1):
    K[i,i] = (alpha[i-1]/l[i-1] + alpha[i]/l[i] +
              beta[i-1]*l[i-1]/3 + beta[i]*l[i]/3)

for i in range(0, N-1):
    K[i+1,i] = -alpha[i]/l[i] + beta[i]*l[i]/6
    K[i,i+1] = K[i+1,i]

# RHS from sources (zero here except BCs)
b[0] = f[0]*l[0]/2
b[N-1] = f[M-1]*l[M-1]/2
for i in range(1, N-1):
    b[i] = f[i-1]*l[i-1]/2 + f[i]*l[i]/2

# === Apply Robin boundary conditions ===
gamma = 1j*k0*np.cos(theta)
g_left = 2j*k0*np.cos(theta)*E0

K[0,0] += gamma
b[0]   += g_left

K[-1,-1] += gamma
# no RHS contribution at right

# === Solve system ===
phi = np.linalg.solve(K, b)

# === Extract R, T from FEM ===
R_num = phi[0]/E0 - 1
T_num = phi[-1]/E0
print("FEM Reflection R =", R_num)
print("FEM Transmission T =", T_num)

# === Analytical solution ===
eta0 = 377.0
kx = k0*np.sin(theta)
kz0 = k0*np.cos(theta)
kzs = k0*np.sqrt(er*ur - (np.sin(theta))**2)
etas = np.sqrt(ur/er)*eta0  # TE wave impedance inside slab

# Fresnel coefficients
r12 = (eta0*kzs - etas*kz0) / (eta0*kzs + etas*kz0)
t12 = 2*eta0*kzs / (eta0*kzs + etas*kz0)

r23 = (etas*kz0 - eta0*kzs) / (etas*kz0 + eta0*kzs)
t23 = 2*etas*kz0 / (etas*kz0 + eta0*kzs)

R_ana = (r12 + r23*np.exp(-2j*kzs*L)) / (1 + r12*r23*np.exp(-2j*kzs*L))
T_ana = (t12*t23*np.exp(-1j*kzs*L)) / (1 + r12*r23*np.exp(-2j*kzs*L))

print("Analytical Reflection R =", R_ana)
print("Analytical Transmission T =", T_ana)

# Field profile (analytical, for plotting)
x_dense = np.linspace(0, L, 2000)
E_analytical = np.zeros_like(x_dense, dtype=complex)

# inside slab: forward + backward
A = (1 + R_ana)
B = R_ana
for i, xval in enumerate(x_dense):
    if xval < 0:
        E_analytical[i] = E0*np.exp(-1j*kz0*xval) + R_ana*E0*np.exp(1j*kz0*xval)
    elif xval <= L:
        # coefficients inside slab from transfer matrix (simplified)
        E_analytical[i] = (t12/(1+r12))*(
            np.exp(-1j*kzs*xval) + r23*np.exp(-1j*kzs*L)*np.exp(1j*kzs*xval))
    else:
        E_analytical[i] = T_ana*E0*np.exp(-1j*kz0*(xval-L))

# === Plot ===
plt.figure(figsize=(10,6))
plt.plot(nodes, phi.real, 'r-', label="FEM Re(E)")
plt.plot(x_dense, E_analytical.real, 'k--', label="Analytical Re(E)")
plt.xlabel("x (m)")
plt.ylabel("E-field")
plt.legend()
plt.grid(True)
plt.show()
