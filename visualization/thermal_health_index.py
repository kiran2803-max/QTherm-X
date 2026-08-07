
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET

# ==========================================
# Load NASA Dataset
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
# Engine 1
# ==========================================

engine = df[df["Engine_ID"] == 1].copy()

# ==========================================
# Thermal Sensors
# ==========================================

thermal_sensors = [
    "Sensor2",
    "Sensor3",
    "Sensor4",
    "Sensor7",
    "Sensor11",
    "Sensor15"
]

# ==========================================
# Normalize Sensors (Min-Max without sklearn)
# ==========================================

normalized = engine[thermal_sensors].copy()

for col in thermal_sensors:
    min_val = normalized[col].min()
    max_val = normalized[col].max()

    normalized[col] = (normalized[col] - min_val) / (max_val - min_val)

# ==========================================
# Thermal Health Index
# ==========================================

engine["THI"] = normalized.mean(axis=1)

# ==========================================
# Save Folder
# ==========================================

os.makedirs("results/nasa", exist_ok=True)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(
    engine["Cycle"],
    engine["THI"],
    linewidth=2
)

plt.title("Thermal Health Index (Engine 1)")

plt.xlabel("Cycle")

plt.ylabel("THI")

plt.grid(True)

plt.savefig(
    "results/nasa/thermal_health_index.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Thermal Health Index generated successfully!")