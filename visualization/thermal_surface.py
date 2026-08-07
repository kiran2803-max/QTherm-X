
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import os
import sys

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

columns = [
    "Engine_ID","Cycle",
    "Setting1","Setting2","Setting3",
    "Sensor1","Sensor2","Sensor3","Sensor4","Sensor5",
    "Sensor6","Sensor7","Sensor8","Sensor9","Sensor10",
    "Sensor11","Sensor12","Sensor13","Sensor14","Sensor15",
    "Sensor16","Sensor17","Sensor18","Sensor19","Sensor20","Sensor21"
]

df.columns = columns

engine = df[df["Engine_ID"] == 1].copy()

thermal_sensors = [
    "Sensor2",
    "Sensor3",
    "Sensor4",
    "Sensor7",
    "Sensor11",
    "Sensor15"
]

# ==========================================
# Normalize
# ==========================================

normalized = engine[thermal_sensors].copy()

for col in thermal_sensors:
    normalized[col] = (
        normalized[col] - normalized[col].min()
    ) / (
        normalized[col].max() - normalized[col].min()
    )

# ==========================================
# Prepare Surface
# ==========================================

X = np.arange(len(engine))

Y = np.arange(len(thermal_sensors))

X, Y = np.meshgrid(X, Y)

Z = normalized.T.values

# ==========================================
# Plot
# ==========================================

fig = plt.figure(figsize=(12,8))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    X,
    Y,
    Z,
    cmap='viridis'
)

ax.set_xlabel("Cycle")
ax.set_ylabel("Sensor")
ax.set_zlabel("Normalized Value")

ax.set_yticks(np.arange(len(thermal_sensors)))
ax.set_yticklabels(thermal_sensors)

plt.title("3D Thermal Sensor Surface")

os.makedirs("results/nasa", exist_ok=True)

plt.savefig(
    "results/nasa/thermal_surface_3d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("3D Thermal Surface Saved!")