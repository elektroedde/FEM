import numpy as np
import pandas as pd
elements = 8
nodes = 9
alpha_x = 1
alpha_y = 1

df = pd.read_csv("elements.csv", header=None)
n = df.iloc[:, 1:].to_numpy(dtype=int)

#Position of nodes
df = pd.read_csv("positions.csv", header=None)
xy = df.iloc[:, 1:].to_numpy(dtype=int)

#b and c
be = np.zeros((elements,3))
ce = np.zeros((elements,3))


for e in range(elements):
    x = xy[n[e,:]-1,0]
    y = xy[n[e,:]-1,1]
    be[e,0] = y[1] - y[2]
    be[e,1] = y[2] - y[0]
    be[e,2] = y[0] - y[1]
    
    ce[e,0] = x[2] - x[1]
    ce[e,1] = x[0] - x[2]
    ce[e,2] = x[1] - x[0]


delta_e = np.zeros(elements)
for i in range(elements):
    delta_e[i] = 1/2 * (be[i,0]*ce[i,1] - be[i,1]*ce[i,0])

K_e = np.zeros((elements,3,3))
for e in range(elements):
    for i in range(3):
        for j in range(3):
            K_e[e,i,j] = 1/(4*delta_e[e])*(alpha_x*be[e,i]*be[e,j] + alpha_y*ce[e,i]*ce[e,j])
