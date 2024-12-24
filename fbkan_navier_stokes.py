#!/bin/env python3

import torch
import torch.nn as nn
import numpy as np
# data imports
from scipy.io import loadmat
# plotting imports
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# aux imports
from time import time

# filter some user warnings from torch broadcast
import warnings
warnings.filterwarnings("ignore")

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define number of collocation points in x and y
Nx = 50
Ny = 50
x = torch.linspace(0, 1, Nx)
y = torch.linspace(0, 1, Ny)

X,Y = torch.meshgrid(x,y, indexing="ij")
# These are the test data, PDE soln.
# Reshape tensors to 2D 
X_test = X.reshape(-1,1)
Y_test = Y.reshape(-1,1)

# Define Reynolds number
Re = 100.0


### Define a generalized coordinate tensor U = [v1, v2, P] ###
ic_U = torch.zeros(size=(Nx,Ny,3)) * 0.
# Top of lid-driven cavity
ic_U[:, -1, 0] = 1. # setting v1 = 1 (lid)
print("ic_U shape", ic_U.shape)


### Boundary Condition Points ###
# Plot boundary conditions
plt.figure()
CP = plt.contourf(X, Y, ic_U[:,:,0], levels=20, cmap='viridis')
plt.title('$v_x$ Velocity Field Contour')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(CP, label='Velocity')
plt.show()

# Left boundary conditions
left_bc_X   = X[[0], :]
left_bc_Y   = Y[[0], :]
left_bc_vx  = ic_U[[0], :, 0]
left_bc_vy  = ic_U[[0], :, 1]
left_bc_p   = ic_U[[0], :, 2]

# Right boundary conditions
right_bc_X  = X[[-1], :]
right_bc_Y  = Y[[-1], :]
right_bc_vx  = ic_U[[-1], :, 0]
right_bc_vy  = ic_U[[-1], :, 1]
right_bc_p   = ic_U[[-1], :, 2]

# Top boundary conditions
top_bc_X  = X[:, [-1]]
top_bc_Y  = Y[:, [-1]]
top_bc_vx  = ic_U[:, [-1], 0]
top_bc_vy  = ic_U[:, [-1], 1]
top_bc_p   = ic_U[:, [-1], 2]

# Bottom boundary conditions
bottom_bc_X   = X[:, [0]]
bottom_bc_Y   = Y[:, [0]]
bottom_bc_vx  = ic_U[:, [0], 0]
bottom_bc_vy  = ic_U[:, [0], 1]
bottom_bc_p   = ic_U[:, [0], 2]

print("Boundary Condition dimensions (l,r,u,d")
print(left_bc_X.shape)
print(right_bc_X.shape)
print(top_bc_X.shape)
print(bottom_bc_X.shape)

print(left_bc_vx.shape)
print(right_bc_vx.shape)
print(top_bc_vx.shape)
print(bottom_bc_vx.shape)

# Then we flatten everything in the order (L R T P)
X_train_bc  = torch.concat([left_bc_X.flatten(), right_bc_X.flatten(), top_bc_X.flatten(), bottom_bc_X.flatten()]).view((-1,1))
Y_train_bc  = torch.concat([left_bc_Y.flatten(), right_bc_Y.flatten(), top_bc_Y.flatten(), bottom_bc_Y.flatten()]).view((-1,1))
vx_train_bc = torch.concat([left_bc_vx.flatten(), right_bc_vx.flatten(), top_bc_vx.flatten(), bottom_bc_vx.flatten()]).view((-1,1))
vy_train_bc = torch.concat([left_bc_vy.flatten(), right_bc_vy.flatten(), top_bc_vy.flatten(), bottom_bc_vy.flatten()]).view((-1,1))
p_train_bc  = torch.concat([left_bc_p.flatten(), right_bc_p.flatten(), top_bc_p.flatten(), bottom_bc_p.flatten()]).view((-1,1))

N_bc = X_train_bc.shape[0]
print(X_train_bc.shape, Y_train_bc.shape, vx_train_bc.shape, "total:", N_bc)


### Colocation Points ###
# Domain bounds
x_lb = X_test[0]
x_ub = X_test[-1]
y_lb = Y_test[0]
y_ub = Y_test[-1]
print(x_lb, x_ub, y_lb, y_ub)

# Choose (N_collocation) collocation Points to Evaluate the PDE
N_cp = 1000
print(f'Number of collocation points for training: {N_cp}')

# Generate collocation points (CP)
X_train_cp = torch.FloatTensor(N_cp, 1).uniform_(float(x_lb), float(x_ub))
Y_train_cp = torch.FloatTensor(N_cp, 1).uniform_(float(y_lb), float(y_ub))

# Append collocation points and boundary points for training
X_train = torch.vstack((X_train_cp, X_train_bc)).float()
Y_train = torch.vstack((Y_train_cp, Y_train_bc)).float()
print(X_train.shape, Y_train.shape)

### Plot Colocation Points for visualization ### 
# Create a figure and axis object
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Scatter plot for CP points
ax.scatter(X_train_cp.detach().cpu().numpy(), Y_train_cp.detach().cpu().numpy(),
           s=4., c='blue', marker='o', label='CP')
# Scatter plot for BC points

ax.scatter(X_train_bc.detach().cpu().numpy(), Y_train_bc.detach().cpu().numpy(),
           s=4., c='red', marker='o', label='BC')

ax.set_title('Sampled BC, and CP (x,t) for training')
ax.set_xlim(x_lb, x_ub)
ax.set_ylim(y_lb, y_ub)
ax.grid(True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right')
ax.set_aspect('equal', 'box')
plt.show()

