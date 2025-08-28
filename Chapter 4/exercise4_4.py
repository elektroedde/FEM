import numpy as np
import pandas as pd

#! INPUT NECESSARY DATA

#? Get total number of nodes
df = pd.read_csv("node_xy.csv")
nodes_xy = df.iloc[:,1:].to_numpy(dtype=float)
N = nodes_xy.shape[0]
print("Total number of nodes: ", N)

#? Get total number of elements
df = pd.read_csv("elements.csv")
elements_index = df.iloc[:,1:].to_numpy(dtype=int)
M = elements_index.shape[0]
print("Total number of elements: ", M)

#? Get values of alpha_x(e), alpha_y(e), f(e) (Temporary all 1)
alpha_x = 1
alpha_y = 1
f = 1
gamma = 0
q = 0

#? Get number of nodes on boundary 1
#? Get global node number nd(i)
#? Get prescribed value of phi(i)

#? Get number of segments on boundary 2
#? Get values of gamma(s) and q(s)
#? Get global node number ns(i)

#! INITIALIZE MATRIX [K] AND {b}

#? Set K(i,j) = 0   i,j = 1 -> N
#? Set b(i) = 0     i = 1 -> N
K = np.zeros((N,N))
b = np.zeros(N)

#! ASSEMBLE ALL TRIANGULAR ELEMENTS IN DOMAIN
b_e = np.zeros(3)
c_e = np.zeros(3)
K_e = np.zeros((3,3))
b_e= np.zeros(3)
for e in range(M):

    i = elements_index[e,0]
    j = elements_index[e,1]
    k = elements_index[e,2]
    b_e[0] = nodes_xy[j,1] - nodes_xy[k,1]
    b_e[1] = nodes_xy[k,1] - nodes_xy[i,1]
    b_e[2] = nodes_xy[i,1] - nodes_xy[j,1]
    c_e[0] = nodes_xy[k,0] - nodes_xy[j,0]
    c_e[1] = nodes_xy[i,0] - nodes_xy[k,0]
    c_e[2] = nodes_xy[j,0] - nodes_xy[i,0]
    delta_e = 1/2*(b_e[0]*c_e[1] - b_e[1]*c_e[0])

    for i in range(3):
        b_e[i] = delta_e/3 * f

        for j in range(3):
            dirac = 0
            if i == j:
                dirac = 1
            
            K_e[i,j] = 1/(4*delta_e)*(alpha_x*b_e[i]*b_e[j] + alpha_y*c_e[i]*c_e[j]) #TODO add beta
            K[elements_index[e,i]-1,elements_index[e,j]-1] += K_e[i, j]

            

#! ASSEMBLE ALL LINE SEGMENTS FOR ROBIN BOUNDARY
df = pd.read_csv("bc.csv")
boundary_condition = df.iloc[:, :].to_numpy(dtype=int)
M_s = boundary_condition.shape[0]
print("Total number of boundary line segments:", M_s)

l_s = np.zeros(M_s)
for s in range(M_s):
    i = boundary_condition[s,0]
    j = boundary_condition[s,1]

    l_s[s] = np.sqrt((nodes_xy[i,0]-nodes_xy[j,0])**2+(nodes_xy[i,1]-nodes_xy[j,1])**2)

    for i in range(2):
        b_i_s = q*l_s[s]/2
        b[boundary_condition[s,i]] += b_i_s
        for j in range(2):
            dirac = 0
            if(i==j):
                dirac = 1
            K_ij_s = gamma*(1+dirac)*l_s[s]/6
            K[boundary_condition[s,i],boundary_condition[s,j]] += K_ij_s

#! IMPOSE DIRICHLET BOUNDARY
df = pd.read_csv("dirichlet.csv")
dirichlet_nodes = df.iloc[:, :].to_numpy(dtype=int)


N_d = dirichlet_nodes.shape[0]
print("Total number of dirichlet nodes:", N_d)

for i in range(N_d):
    for j in range(N):
        if(j == dirichlet_nodes[i,0]):
            print("YEA")
            K[j,j] = 1
            b[j] = dirichlet_nodes[i,1]
        else:
            b[j] -= K[j,dirichlet_nodes[i,0]]*dirichlet_nodes[i,1]
            K[dirichlet_nodes[i,0],j] = 0
            K[j,dirichlet_nodes[i,0]] = 0

phi = np.linalg.solve(K, b)

import matplotlib.pyplot as plt
grid = np.zeros((3,3))
for col in range(3):
    for row in range(3):
        idx = col*3 + row
        grid[row, col] = phi[idx]

grid_plot = np.flipud(grid)

fig, ax = plt.subplots(figsize=(16,9))
im = ax.imshow(grid_plot, origin='lower', interpolation='quadric',cmap="jet")

ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xticklabels(["x=0","x=1","x=2"])
ax.set_yticklabels(["y=0","y=1","y=2"])
ax.set_title("FEM solution φ on 3×3 node grid")


plt.colorbar(im, ax=ax, label="φ value")
plt.tight_layout()
plt.show()