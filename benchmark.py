import time
from xml.parsers.expat import model
import torch
import csv
import os

QUBIT_LIST = [2, 3, 4]
N_LAYERS = 2

from models.classical_pinn import PINN
from models.hybrid_pinn import HybridPINN

from physics.heat_equation import (
    compute_heat_residual,
    compute_ic_loss,
    compute_bc_loss
)

from evaluation.metrics import (
    compute_mae,
    compute_mse,
    compute_relative_l2_error
)
# ==========================================
# Device
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("QTherm-X Benchmark")
print("=" * 50)
print("Running on :", device)
# ==========================================
# Training Points
# ==========================================

N = 5000

x = torch.rand(N, 1, requires_grad=True)
t = torch.rand(N, 1, requires_grad=True)

X_train = torch.cat([x, t], dim=1).to(device)

print("Training Points :", X_train.shape)
# ==========================================
# Train Function
# ==========================================

def train_model(model, model_name, epochs=3000, lr=0.001):

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

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
        total_loss = physics_loss + ic_loss + bc_loss

        total_loss.backward()

        optimizer.step()

        # Print training progress every 500 epochs
        if epoch % 500 == 0:
            print(
                f"{model_name} | Epoch {epoch}/{epochs} | "
                f"Loss = {total_loss.item():.6e}"
            )

    training_time = time.time() - start_time

    num_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # Relative L2 Error
    relative_l2 = compute_relative_l2_error(model, X_train)
    mae = compute_mae(model, X_train)

    mse = compute_mse(model, X_train)

    return {
    "Model": model_name,
    "PDE Loss": physics_loss.item(),
    "IC Loss": ic_loss.item(),
    "BC Loss": bc_loss.item(),
    "Total Loss": total_loss.item(),
    "MAE": mae,
    "MSE": mse,
    "Relative L2": relative_l2,
    "Training Time": training_time,
    "Parameters": num_parameters
}
# ==========================================
# Save Benchmark Results
# ==========================================

def save_results(classical_results, hybrid_results):

    os.makedirs("results/benchmark", exist_ok=True)

    file_path = "results/benchmark/results.csv"

    file_exists = os.path.exists(file_path)

    with open(file_path, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        if not file_exists:
          writer.writerow([
          "Model",
          "Qubits",
          "Layers",
          "PDE Loss",
          "IC Loss",
          "BC Loss",
          "Total Loss",
          "MAE",
          "MSE",
          "Relative L2",
          "Training Time",
          "Parameters"
        ])

        all_results = [classical_results] + hybrid_results

        for result in all_results:

          writer.writerow([
          result["Model"],
          result["Qubits"],
          result["Layers"],
          result["PDE Loss"],
          result["IC Loss"],
          result["BC Loss"],
          result["Total Loss"],
          result["MAE"],
          result["MSE"],
          result["Relative L2"],
          result["Training Time"],
          result["Parameters"]
            ])
# ==========================================
# Benchmark Starts
# ==========================================

print("\nTraining Classical PINN...\n")

classical_results = train_model(
    PINN(),
    "Classical PINN"
)
classical_results["Qubits"] = "-"
classical_results["Layers"] = "-"

print("Done!\n")

hybrid_results = []

for qubits in QUBIT_LIST:

    print(f"Training Hybrid QAPINN ({qubits} Qubits)...\n")

    result = train_model(
        HybridPINN(n_qubits=qubits, n_layers=N_LAYERS),
        f"Hybrid QAPINN ({qubits}Q)"
    )

    result["Qubits"] = qubits
    result["Layers"] = N_LAYERS

    hybrid_results.append(result)

    print("Done!\n")

# ==========================================
# Results
# ==========================================

print("=" * 120)
print("QTHERM-X BENCHMARK RESULTS")
print("=" * 120)

print(
    f"{'Model':<25}"
    f"{'Qubits':<10}"
    f"{'PDE Loss':<15}"
    f"{'IC Loss':<15}"
    f"{'BC Loss':<15}"
    f"{'Total Loss':<15}"
    f"{'MAE':<15}"
    f"{'MSE':<15}"
    f"{'Rel L2':<15}"
)

print("-" * 120)

all_results = [classical_results] + hybrid_results

for result in all_results:

    print(
        f"{result['Model']:<25}"
        f"{str(result['Qubits']):<10}"
        f"{result['PDE Loss']:<15.6e}"
        f"{result['IC Loss']:<15.6e}"
        f"{result['BC Loss']:<15.6e}"
        f"{result['Total Loss']:<15.6e}"
        f"{result['MAE']:<15.6e}"
        f"{result['MSE']:<15.6e}"
        f"{result['Relative L2']:<15.6e}"
    )

print("=" * 120)
save_results(classical_results, hybrid_results)

print("\nBenchmark saved to results/benchmark/results.csv")