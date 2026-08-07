import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET
# ==========================================
# Load Dataset
# ==========================================

file_path = f"datasets/NASA_CMAPSS/train_{DATASET}.txt"

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None
)

# ==========================================
# Column Names
# ==========================================

columns = [
    "Engine_ID","Cycle",
    "Setting1","Setting2","Setting3",
    "Sensor1","Sensor2","Sensor3","Sensor4","Sensor5",
    "Sensor6","Sensor7","Sensor8","Sensor9","Sensor10",
    "Sensor11","Sensor12","Sensor13","Sensor14","Sensor15",
    "Sensor16","Sensor17","Sensor18","Sensor19","Sensor20","Sensor21"
]

df.columns = columns

# ==========================================
# Compute RUL
# ==========================================

max_cycle = df.groupby("Engine_ID")["Cycle"].max()

df["RUL"] = (
    max_cycle[df["Engine_ID"]].values
    - df["Cycle"]
)

# ==========================================
# Plot Engine 1
# ==========================================

engine = df[df["Engine_ID"] == 1]

os.makedirs("results/nasa", exist_ok=True)

plt.figure(figsize=(8,5))

plt.plot(
    engine["Cycle"],
    engine["RUL"],
    linewidth=2
)

plt.title("Engine 1 Remaining Useful Life")

plt.xlabel("Cycle")

plt.ylabel("RUL")

plt.grid(True)

import os

save_path = os.path.abspath("results/nasa/engine1_rul.png")

plt.savefig(save_path, dpi=300)

print(f"Saved to: {save_path}")

plt.show()

print("Graph Saved Successfully!")
# ==========================================
# Multiple Engine RUL Curves
# ==========================================

# ==========================================
# Multiple Engine RUL Curves
# ==========================================

plt.figure(figsize=(10, 6))

colors = ["blue", "red", "green", "orange", "purple"]

for i, engine_id in enumerate([1, 2, 3, 4, 5]):

    engine = df[df["Engine_ID"] == engine_id]

    plt.plot(
        engine["Cycle"],
        engine["RUL"],
        color=colors[i],
        linewidth=2,
        label=f"Engine {engine_id}"
    )

plt.title("Remaining Useful Life of Engines 1-5")
plt.xlabel("Cycle")
plt.ylabel("RUL")
plt.legend()
plt.grid(True)

os.makedirs("results/nasa", exist_ok=True)

save_path = "results/nasa/multiple_engine_rul.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")

print("Saved:", os.path.abspath(save_path))

plt.show()