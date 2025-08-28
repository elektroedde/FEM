import numpy as np
import matplotlib.pyplot as plt


M = 30         # number of elements
L = 3/2           # total length
N = M + 1

# Uniform element length
le = L / M

# Per-element data (uniform example). Use lists so you can set element-wise.
alpha = [9.0] * M   # stiffness coefficient a^e
beta  = [-np.pi**2] * M   # reaction coefficient b^e
f     = [0.0] * M   # source term f^e (could be function or constant)

l     = [le] * M    # element lengths l^e


# Build node coordinates and element connectivity
nodes = np.linspace(0.0, L, N)


# connectivity for linear elements: element e connects node e and node e+1
elements = [(e, e+1) for e in range(M)]

gamma = 0.0
p = -1.0
q = 3

print("=== FEM input data ===")
print(f"Number of elements M = {M}")
print(f"Total length L = {L}")
print(f"Number of nodes = {N}")
print(f"Element length (uniform) = {le:.6f}")
print()

print("Nodes (index : coordinate):")
for i,x in enumerate(nodes):
    print(f"  {i:2d} : {x:.6f}")
print()

print("Elements (index : node_i -> node_j , length , alpha, beta, f):")
for e,(n1,n2) in enumerate(elements):
    print(f"  e={e:2d} : {n1:2d} -> {n2:2d} , l^e={l[e]:.6f}, alpha^e={alpha[e]:.3g}, beta^e={beta[e]:.3g}, f^e={f[e]:.3g}")
print()



K = np.zeros((N, N))  # global stiffness matrix

K[0,0] = alpha[0]/l[0] + beta[0]*l[0]/3

K[N-1, N-1] = alpha[M-1]/l[M-1] + beta[M-1]*l[M-1]/3 + gamma

for i in range(1,N-1):
    K[i,i] = alpha[i-1]/l[i-1] + alpha[i]/l[i] + beta[i-1]*l[i-1]/3 + beta[i]*l[i]/3


for i in range(0,N-1):
    K[i+1,i] = -alpha[i]/l[i] + beta[i]*l[i]/6
    K[i,i+1] = K[i+1,i]

# K is correct
b = np.zeros(N)

b[0] = f[0]*l[0]/2
b[N-1] = f[M-1]*l[M-1]/2 + q

for i in range(1,N-1):
    b[i] = f[i-1]*l[i-1]/2 + f[i]*l[i]/2

# b is correct



K[0,0] = 1.0
b[0] = p
for i in range(1,N):
    K[0, i] = 0.0


K[N-1,N-1] = 1.0
for i in range(0,N-1):
    K[N-1,i] = 0.0



for i in range(1,N):
    b[i] = b[i] - K[i,0]*p




a = np.array(np.diag(K))

c = np.array(np.diag(K,1))

for i in range(1, N-1):
    a[i] = a[i] - c[i-1]**2/a[i-1]
    b[i] = b[i] - c[i-1]*b[i-1]/a[i-1]


print(a)
print(b)

phi = np.zeros(N)
phi[N-1] = b[N-1]/a[N-1]


for i in range(N-2, -1, -1):
    phi[i] = (b[i] - c[i]*phi[i+1])/a[i]

print(phi)

x = np.linspace(0.0, L, M)


# Analytical solution
y = 3*np.sin(np.pi*x/3) + - np.cos(np.pi*x/3)

plt.plot(nodes, phi, marker='o')
plt.plot(x, y, marker='x')
plt.show()