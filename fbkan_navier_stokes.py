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

print("Boundary Condition dimensions (l,r,u,d)")
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


### Dataset Creation for Neuromancer ###
from neuromancer.dataset import DictDataset

# turn on gradients for PINN
X_train.requires_grad=True
Y_train.requires_grad=True

# Training dataset
train_data = DictDataset({'x': X_train, 'y':Y_train}, name='train')
# test dataset
test_data = DictDataset({'x': X_test, 'y':Y_test}, name='test')

# torch dataloaders
batch_size = X_train.shape[0]  # full batch training
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           collate_fn=train_data.collate_fn,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         collate_fn=test_data.collate_fn,
                                         shuffle=False)




### Neuromancer NN Architecture ###
from neuromancer.modules import blocks
from neuromancer.system import Node, System

# neural net to solve the PDE problem
net = blocks.KANBlock(insize=2,
                      outsize=3,
                      hsizes=[5,5],
                      spline_order=5)

# symbolic wrapper of the neural net
# [x,y] inputs, [U] generalized outputs [vx, vy, p]
pde_net = Node(net, ['x', 'y'], ['U'], name='net')
print("symbolic inputs  of the pde_net:", pde_net.input_keys)
print("symbolic outputs of the pde_net:", pde_net.output_keys)

#pde_net(train_data.datadict) gives us our U tensor.
# forward pass
net_out = pde_net(train_data.datadict)
print(pde_net(train_data.datadict).keys())
print(pde_net(train_data.datadict)['U'].shape)


### Define Variables for the NN and Optimization Problems ###
from neuromancer.constraint import variable

# symbolic Neuromancer variables
U = variable('U')
vx = variable('U')[:,[0]] # All timesteps, 1st dim
vy = variable('U')[:,[1]]
p = variable('U')[:,[2]]
x = variable('x')  # spatial coordinate 1
y = variable('y')  # spatial coordinate 2

# get the symbolic derivatives
dvx_dx   = vx.grad(x)
dvy_dx   = vy.grad(x)
dp_dx    = p.grad(x)

dvx_dy   = vx.grad(y)
dvy_dy   = vy.grad(y)
dp_dy    = p.grad(y)

d2vx_dx2 = dvx_dx.grad(x)
d2vy_dx2 = dvy_dx.grad(x)
d2vx_dy2 = dvx_dy.grad(y)
d2vy_dy2 = dvy_dy.grad(y)

# get the PINN form
f_pinn_1 = (1./Re)*(d2vx_dx2 + d2vx_dy2) - vx * dvx_dx - vy * dvx_dy - dp_dx
f_pinn_2 = (1./Re)* (d2vy_dx2 + d2vy_dy2) - vy * dvy_dx - vy * dvy_dy - dp_dy
f_pinn_3 = dvx_dx + dvy_dy

# Take a moment to just visualize the PINNs
print("PINN 1")
f_pinn_1.show()
print("PINN 2")
f_pinn_2.show()
print("PINN 3")
f_pinn_3.show()


# check the shapes of the forward pass of the symbolic PINN terms
## (imtweakinimtweakinimtweakinimtweakin) ##
print(f"vx: {vx({**net_out, **train_data.datadict}).shape}")
print(f"vy: {vy({**net_out, **train_data.datadict}).shape}")
print(f"p: {p({**net_out, **train_data.datadict}).shape}")

print(f"dvx_dx: {dvx_dx({**net_out, **train_data.datadict}).shape}")
print(f"dvy_dx: {dvy_dx({**net_out, **train_data.datadict}).shape}")
print(f"dp_dx: {dp_dx({**net_out, **train_data.datadict}).shape}")

print(f"dvx_dy: {dvx_dy({**net_out, **train_data.datadict}).shape}")
print(f"dvy_dy: {dvy_dy({**net_out, **train_data.datadict}).shape}")
print(f"dp_dy: {dp_dy({**net_out, **train_data.datadict}).shape}")

print(f"d2vx_dx2: {d2vx_dx2({**net_out, **train_data.datadict}).shape}")
print(f"d2vy_dx2: {d2vy_dx2({**net_out, **train_data.datadict}).shape}")
print(f"d2vx_dy2: {d2vx_dy2({**net_out, **train_data.datadict}).shape}")
print(f"d2vy_dy2: {d2vy_dy2({**net_out, **train_data.datadict}).shape}")

print(f"f_pinn_1: {f_pinn_1({**net_out, **train_data.datadict}).shape}")
print(f"f_pinn_2: {f_pinn_2({**net_out, **train_data.datadict}).shape}")
print(f"f_pinn_3: {f_pinn_3({**net_out, **train_data.datadict}).shape}")



### Loss Functions ###
# Now is the time to design our fulcrum, with which we shall move the world.

# Idea is to use PDE residuals for each CP
#   (essentially use automatic differentiation (autograd) to just calc PDE of these Colocation points).
# then use that in a weighted sum with the BP,
# Where BP is supervised and predicted by the 3 NN's (V_x, V_y, P)

# scaling factor for better convergence
scaling_cp = 1.
scaling_bc = 1.
scaling_continuity = 1.

# PDE CP loss (MSE)
l_cp_1 = scaling_cp*(f_pinn_1 == 0.)^2
l_cp_2 = scaling_cp*(f_pinn_2 == 0.)^2
l_cp_3 = scaling_continuity*(f_pinn_3 == 0.)^2
l_cp_1.update_name("loss_cp_1")
l_cp_2.update_name("loss_cp_2")
l_cp_3.update_name("loss_cp_3")
  
# PDE BC loss (MSE)
# note: remember that we concatenated CP and BC
l_bc_1 = scaling_bc*( vx[-N_bc:] == vx_train_bc)^2
l_bc_2 = scaling_bc*( vy[-N_bc:] == vy_train_bc)^2
l_bc_3 = scaling_bc*( p[-N_bc:] == p_train_bc)^2

l_bc_1.update_name("loss_bc_1")
l_bc_2.update_name("loss_bc_2")
l_bc_3.update_name("loss_bc_3")



### Setup the Constraint Optimization Problem  ### 
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

# we do so by using the PenaltyLoss as our Loss Function
pinn_loss = PenaltyLoss(objectives=[l_cp_1, l_cp_2, l_cp_3, l_bc_1, l_bc_2, l_bc_3], constraints=[])

# and then construct the PINN optimization problem
problem = Problem(nodes=[pde_net],      # list of nodes (neural nets) to be optimized
                  loss=pinn_loss,       # physics-informed loss function
                  grad_inference=True   # argument for allowing computation of gradients at the inference time
                 )

### Train the Network ###
from neuromancer.trainer import Trainer

optimizer = torch.optim.AdamW(problem.parameters(), lr=1e-2)
epochs = 20000

#  Neuromancer trainer
trainer = Trainer(
    problem.to(device),
    train_loader,
    optimizer=optimizer,
    epochs=epochs,
    epoch_verbose=500,
    train_metric='train_loss',
    dev_metric='train_loss',
    eval_metric="train_loss",
    warmup=epochs,
    device=device
)
# Train PINN
t0 = time()
print("Beginning KAN NN training now...")
best_model = trainer.train()
print(f"Elapsed time: {time()-t0} s")

### Model has been trained. Now use the best. ###
problem.load_state_dict(best_model)


### Evaluate Results ### 
# Evaluate trained PINN on test data (all the data in the domain)
PINN = problem.nodes[0].cpu()
U_pinn = PINN(test_data.datadict)['U']

# arrange data for plotting
vx_pinn = U_pinn[:,0].reshape(shape=[Nx,Ny]).detach().cpu()
vy_pinn = U_pinn[:,1].reshape(shape=[Nx,Ny]).detach().cpu()
p_pinn = U_pinn[:,2].reshape(shape=[Nx,Ny]).detach().cpu()

# plot PINN solution
plt.figure(figsize=(20, 5))

# # Plot for the second heatmap (PINN solution)
ax2 = plt.subplot(1, 3, 1)
CP2 = plt.contourf(X, Y, vx_pinn, levels=20, cmap='viridis')
plt.title('PIKAN solution, $v_x$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar(CP2, label='$v_x$')
ax2.set_aspect('equal', adjustable='box')

ax2 = plt.subplot(1, 3, 2)
CP2 = plt.contourf(X, Y, vy_pinn, levels=20, cmap='viridis')
plt.title('PIKAN solution, $v_y$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar(CP2, label='$v_y$')
ax2.set_aspect('equal', adjustable='box')

ax2 = plt.subplot(1, 3, 3)
CP2 = plt.contourf(X, Y, p_pinn, levels=20, cmap='viridis')
plt.title('PIKAN solution, $p$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.colorbar(CP2, label='$p$')
ax2.set_aspect('equal', adjustable='box')





### Extra: Velocity Magnitude and Streamline Comparisons ### 
# Plot
fig = plt.figure(figsize=(9, 5))
gs = GridSpec(1, 2, width_ratios=[1, 0.8], height_ratios=[1])

# Left plot: PINN solution with velocity magnitude
ax1 = fig.add_subplot(gs[0, 0])
CP1 = ax1.contourf(X, Y, np.sqrt(vx_pinn**2 + vy_pinn**2), levels=20, cmap='viridis')
ax1.set_title('PIKAN solution, $|\\boldsymbol{v}|$', fontsize=14)
ax1.set_xlabel('$x$', fontsize=12)
ax1.set_ylabel('$y$', fontsize=12)
plt.colorbar(CP1, ax=ax1, label='$|\\boldsymbol{v}|$')
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim(0., 1.)
ax1.set_ylim(0., 1.)

# Right plot: Streamlines without magnitude colorbar
ax2 = fig.add_subplot(gs[0, 1])
x_ = np.linspace(0, 1, Nx)
y_ = np.linspace(0, 1, Ny)
strm = ax2.streamplot(x_, y_, vx_pinn.T, vy_pinn.T, color='k', density=2, minlength=0.1)
ax2.set_title('Streamlines', fontsize=14)
ax2.set_xlabel('$x$', fontsize=12)
ax2.set_ylabel('$y$', fontsize=12)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(0., 1.)
ax2.set_ylim(0., 1.)

# Apply tight layout for even spacing
plt.tight_layout()
plt.show()



### Extra: Centerline Analysis ###
# Get centerline solutions
x_ = test_data.datadict['x'].reshape(Nx,Ny).cpu()
y_ = test_data.datadict['y'].reshape(Nx,Ny).cpu()
eval_solution_pinn = PINN(test_data.datadict)['U'].reshape(Nx,Ny,-1)
vx_est = eval_solution_pinn[:,:,0].cpu().detach().numpy()
vy_est = eval_solution_pinn[:,:,1].cpu().detach().numpy()

# Reference velocities (Ghia et al., 1982)
ref_x = torch.tensor([1.00000,0.9688,0.9609,0.9531,0.9453,0.9063,0.8594,0.8047,0.5000,0.2344,0.2266,0.1563,0.0938,0.0781,0.0703,0.0625,0.0000])
ref_y = torch.tensor([1.0000,0.9766,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,0.5000,0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0000])
ref_vx_Re100 = torch.tensor([1.00000,0.84123,0.78871,0.73722,0.68717,0.23151,0.00332,-0.13641,-0.20581,-0.21090,-0.15662,-0.10150,-0.06434,-0.04775,-0.04192,-0.03717,0.00000])
ref_vy_Re100 = torch.tensor([0.00000,-0.05906,-0.07391,-0.08864,-0.10313,-0.16914,-0.22445,-0.24533,0.05454,0.17527,0.17507,0.16077,0.12317,0.10890,0.10091,0.09233,0.00000])

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

# x vs vy - Plotting vertical centerline comparison
ax[0].scatter(ref_x, ref_vy_Re100, color='blue', marker='o', edgecolor='black', label='Ref. (Ghia et al., 1982)')
ax[0].plot(x_[:, Nx // 2], vy_est[:, Ny // 2], color='black', linestyle='--', linewidth=1.5, label='PIKAN')
ax[0].set_xlabel('$x$', fontsize=12)
ax[0].set_ylabel('$v_y$', fontsize=12)
ax[0].set_title("Vertical Centerline: $v_y$ vs $x$", fontsize=14)
ax[0].grid(True, linestyle='--', alpha=0.6)
ax[0].legend()

# vx vs y - Plotting horizontal centerline comparison
ax[1].scatter(ref_vx_Re100, ref_y, color='red', marker='o', edgecolor='black', label='Ref. (Ghia et al., 1982)')
ax[1].plot(vx_est[Nx // 2, :], y_[Nx // 2, :], color='black', linestyle='--', linewidth=1.5, label='PIKAN')
ax[1].set_xlabel('$v_x$', fontsize=12)
ax[1].set_ylabel('$y$', fontsize=12)
ax[1].set_title("Horizontal Centerline: $v_x$ vs $y$", fontsize=14)
ax[1].grid(True, linestyle='--', alpha=0.6)
ax[1].legend()

plt.tight_layout()
plt.show()

