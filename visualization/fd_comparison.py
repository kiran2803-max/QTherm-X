import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Dataset Information
# -----------------------------

comparison = pd.DataFrame({
    "Metric": [
        "Number of Engines",
        "Number of Samples",
        "Operating Conditions",
        "Fault Modes",
        "Maximum RUL"
    ],
    "FD001": [
        100,
        20631,
        1,
        1,
        191
    ],
    "FD004": [
        249,
        61249,
        6,
        2,
        320
    ]
})

# -----------------------------
# Save CSV
# -----------------------------

os.makedirs("results/comparison", exist_ok=True)

comparison.to_csv(
    "results/comparison/fd001_vs_fd004_summary.csv",
    index=False
)

print("Summary CSV Saved!")

# -----------------------------
# Number of Samples
# -----------------------------

plt.figure(figsize=(6,5))

plt.bar(
    ["FD001","FD004"],
    [20631,61249]
)

plt.title("NASA Dataset Size Comparison")

plt.ylabel("Samples")

plt.savefig(
    "results/comparison/dataset_size.png",
    dpi=300
)

plt.close()

# -----------------------------
# Number of Engines
# -----------------------------

plt.figure(figsize=(6,5))

plt.bar(
    ["FD001","FD004"],
    [100,249]
)

plt.title("Number of Engines")

plt.ylabel("Engines")

plt.savefig(
    "results/comparison/engine_count.png",
    dpi=300
)

plt.close()

# -----------------------------
# Maximum RUL
# -----------------------------

plt.figure(figsize=(6,5))

plt.bar(
    ["FD001","FD004"],
    [191,320]
)

plt.title("Maximum Remaining Useful Life")

plt.ylabel("Cycles")

plt.savefig(
    "results/comparison/max_rul.png",
    dpi=300
)

plt.close()

print("Comparison Graphs Saved!")