import numpy as np
import matplotlib.pyplot as plt

# 5 wavelength thick dielectric slab
# er = 4.0 - j0.1
# ur = 2.0
freq = 1e9
wavelength = 3e8/freq
M = 5000         # number of elements
L = 5*wavelength           # total length
N = M + 1
# Uniform element length
le = L / M
er = 4.0 - 1j*0.1
ur = 2.0
k0 = 2*np.pi/wavelength
E0 = 1.0


theta = 0
theta = np.deg2rad(theta)
# Per-element data (uniform example). Use lists so you can set element-wise.
alpha = [1.0/ur] * M   # stiffness coefficient a^e
beta  = [-k0**2*(er - np.sin(theta)**2/ur)] * M   # reaction coefficient b^e
f     = [0.0] * M   # source term f^e (could be function or constant)
l     = [le] * M    # element lengths l^e
# Build node coordinates and element connectivity
nodes = np.linspace(0.0, L, N)
# connectivity for linear elements: element e connects node e and node e+1
elements = [(e, e+1) for e in range(M)]
gamma = 1j*k0*np.cos(theta)
p = 0
q = 2j*k0*np.cos(theta)*E0*np.exp(1j*k0*L*np.cos(theta))
K = np.zeros((N, N), dtype=np.complex128)  # global stiffness matrix
K[0,0] = alpha[0]/l[0] + beta[0]*l[0]/3
K[N-1, N-1] = alpha[M-1]/l[M-1] + beta[M-1]*l[M-1]/3
for i in range(1,N-1):
    K[i,i] = alpha[i-1]/l[i-1] + alpha[i]/l[i] + beta[i-1]*l[i-1]/3 + beta[i]*l[i]/3
for i in range(0,N-1):
    K[i+1,i] = -alpha[i]/l[i] + beta[i]*l[i]/6
    K[i,i+1] = K[i+1,i]
# K is correct
b = np.zeros(N, dtype=np.complex128) 
b[0] = f[0]*l[0]/2
b[N-1] = f[M-1]*l[M-1]/2
for i in range(1,N-1):
    b[i] = f[i-1]*l[i-1]/2 + f[i]*l[i]/2
# b is correct
#Modify KNN and bN
K[N-1,N-1] += gamma
b[N-1] += q
K[0,0] = 1
b[0] = p
for i in range(1,N):
    K[0, i] = 0.0
for i in range(1,N):
    b[i] = b[i] - K[i,0]*p
for i in range(1,N):
    K[i, 0] = 0.0
a = np.array(np.diag(K), dtype=np.complex128)
c = np.array(np.diag(K,1), dtype=np.complex128)
for i in range(1, N-1):
    a[i] = a[i] - c[i-1]**2/a[i-1]
    b[i] = b[i] - c[i-1]*b[i-1]/a[i-1]
phi = np.zeros(N, dtype=np.complex128)
phi[N-1] = b[N-1]/a[N-1]
for i in range(N-2, -1, -1):
    phi[i] = (b[i] - c[i]*phi[i+1])/a[i]
    

plt.plot(nodes, np.real(phi), label='Re(E)')
plt.show()