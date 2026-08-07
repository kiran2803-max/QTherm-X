import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Load Benchmark Results
# ==========================================

csv_file = "results/benchmark/results.csv"

df = pd.read_csv(csv_file)

# Remove duplicates if any
df = df.drop_duplicates(subset=["Model"], keep="last")

# Create folder
os.makedirs("results/benchmark/figures", exist_ok=True)

# Metrics to plot
metrics = [
    ("Total Loss", "Total Loss"),
    ("PDE Loss", "PDE Loss"),
    ("MAE", "MAE"),
    ("MSE", "MSE"),
    ("Relative L2", "Relative L2"),
    ("Training Time", "Training Time"),
    ("Parameters", "Parameters")
]
for title, column in metrics:

    plt.figure(figsize=(7,5))

    plt.bar(df["Model"], df[column])

    plt.xticks(rotation=15)

    plt.ylabel(title)

    plt.title(title + " Comparison")

    plt.tight_layout()

    plt.savefig(
        f"results/benchmark/figures/{column}.png",
        dpi=300
    )

    plt.close()

print("All benchmark figures generated successfully.")