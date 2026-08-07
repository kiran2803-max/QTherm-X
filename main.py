"""
===========================================
QTherm-X
Main Program

Classical Physics-Informed Neural Network
for solving the 1D Heat Equation

Version : 1.0
===========================================
"""

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.hybrid_pinn import HybridPINN 
from physics.heat_equation import (
    compute_heat_residual,
    compute_ic_loss,
    compute_bc_loss
)
from visualization.plots import plot_loss, plot_heatmap, plot_3d_surface
from utils.hotspot import detect_hotspot
# ==========================================
# DEVICE CONFIGURATION
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("QTherm-X Started")
print("=" * 50)
print(f"Running on : {device}")

# ==========================================
# RANDOM SEED
# ==========================================

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# CREATE MODEL
# ==========================================

model = HybridPINN().to(device)

print("\nModel Created Successfully!\n")

print(model)

# ==========================================
# GENERATE TRAINING DATA
# ==========================================

N = 5000

x = np.random.rand(N, 1)

t = np.random.rand(N, 1)

X = np.hstack((x, t))

X_train = torch.tensor(
    X,
    dtype=torch.float32,
    requires_grad=True
).to(device)

print("\nTraining Points :", X_train.shape)

# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 3000

loss_history = []

print("\nSetup Complete!")
print("Ready to Train...\n")
# ==========================================
# TRAINING LOOP
# ==========================================

print("=" * 50)
print("Training Started...")
print("=" * 50)

for epoch in range(epochs):

    optimizer.zero_grad()

   # Physics Loss
residual = compute_heat_residual(model, X_train)
physics_loss = torch.mean(residual ** 2)

# Initial Condition Loss
ic_loss = compute_ic_loss(model)

    # Boundary Condition Loss
bc_loss = compute_bc_loss(model)

    # Total Loss
loss = physics_loss + ic_loss + bc_loss

    # Backpropagation
loss.backward()

    # Update model parameters
optimizer.step()

    # Store loss
loss_history.append(loss.item())

    # Print progress
if epoch % 200 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"PDE = {physics_loss.item():.8f} | "
            f"IC = {ic_loss.item():.8f} | "
            f"BC = {bc_loss.item():.8f} | "
            f"Total = {loss.item():.8f}"
        )

print("\nTraining Finished Successfully!")

# ==========================================
# SAVE MODEL
# ==========================================

torch.save(model.state_dict(), "results/model.pth")

print("Model saved to results/model.pth")
# ==========================================
# VISUALIZATION
# ==========================================

plot_loss(loss_history)

plot_heatmap(model, device)

plot_3d_surface(model, device)

detect_hotspot(model, device)
print("\nVisualization Complete!")