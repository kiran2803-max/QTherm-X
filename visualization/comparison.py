import matplotlib.pyplot as plt

# ==========================================
# Benchmark Results
# (Replace these with new values after every benchmark)
# ==========================================

classical = {
    "PDE Loss": 0.00026954,
    "Relative L2": 0.02006644,
    "Training Time": 169.07,
    "Parameters": 8577
}

hybrid = {
    "PDE Loss": 0.00014232,
    "Relative L2": 0.01148775,
    "Training Time": 1555.61,
    "Parameters": 1273
}

# ==========================================
# Figure 1 : PDE Loss
# ==========================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Classical PINN", "Hybrid QAPINN"],
    [classical["PDE Loss"], hybrid["PDE Loss"]]
)

plt.ylabel("PDE Loss")
plt.title("PDE Residual Comparison")

plt.tight_layout()
plt.savefig("results/benchmark/pde_loss_comparison.png")
plt.close()

# ==========================================
# Figure 2 : Relative L2 Error
# ==========================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Classical PINN", "Hybrid QAPINN"],
    [classical["Relative L2"], hybrid["Relative L2"]]
)

plt.ylabel("Relative L2 Error")
plt.title("Relative L2 Comparison")

plt.tight_layout()
plt.savefig("results/benchmark/l2_comparison.png")
plt.close()

# ==========================================
# Figure 3 : Training Time
# ==========================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Classical PINN", "Hybrid QAPINN"],
    [classical["Training Time"], hybrid["Training Time"]]
)

plt.ylabel("Seconds")
plt.title("Training Time Comparison")

plt.tight_layout()
plt.savefig("results/benchmark/training_time.png")
plt.close()

# ==========================================
# Figure 4 : Parameters
# ==========================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Classical PINN", "Hybrid QAPINN"],
    [classical["Parameters"], hybrid["Parameters"]]
)

plt.ylabel("Trainable Parameters")
plt.title("Parameter Comparison")

plt.tight_layout()
plt.savefig("results/benchmark/parameter_comparison.png")
plt.close()

print("=" * 50)
print("Comparison Graphs Generated Successfully!")
print("=" * 50)