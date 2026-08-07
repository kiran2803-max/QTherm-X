import matplotlib.pyplot as plt
import numpy as np


def plot_loss(loss_history):

    plt.figure(figsize=(8,5))

    plt.plot(loss_history, linewidth=2)

    plt.title("PINN Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Physics Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("results/loss_curve.png")

    plt.close()

    print("Loss Curve Saved!")


def plot_heatmap(model, device):

    from mpl_toolkits.mplot3d import Axes3D


def plot_3d_surface(model, device):

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)

    X, T = np.meshgrid(x, t)

    grid = np.column_stack((X.flatten(), T.flatten()))

    import torch

    grid_tensor = torch.tensor(
        grid,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        prediction = model(grid_tensor)

    Z = prediction.cpu().numpy().reshape(100, 100)

    fig = plt.figure(figsize=(10,7))

    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X,
        T,
        Z,
        cmap="jet",
        edgecolor="none"
    )

    ax.set_title("3D Temperature Distribution")

    ax.set_xlabel("Position")

    ax.set_ylabel("Time")

    ax.set_zlabel("Temperature")

    plt.tight_layout()

    plt.savefig("results/temperature_surface_3d.png")

    plt.close()

    print("3D Surface Saved!")

    x = np.linspace(0,1,100)

    t = np.linspace(0,1,100)

    X,T = np.meshgrid(x,t)

    grid = np.column_stack((X.flatten(),T.flatten()))

    import torch

    grid_tensor = torch.tensor(
        grid,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        prediction = model(grid_tensor)

    temperature = prediction.cpu().numpy().reshape(100,100)

    plt.figure(figsize=(8,6))

    plt.imshow(
        temperature,
        origin="lower",
        extent=[0,1,0,1],
        aspect="auto"
    )

    plt.colorbar(label="Temperature")

    plt.xlabel("Position")

    plt.ylabel("Time")

    plt.title("Temperature Distribution")

    plt.tight_layout()

    plt.savefig("results/temperature_heatmap.png")

    plt.close()

    print("Heatmap Saved!")