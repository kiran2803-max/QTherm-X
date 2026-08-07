
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET

# ==========================================
# Load Dataset
# ==========================================

file_path = "datasets/NASA_CMAPSS/train_FD001.txt"

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
# Create Output Folder
# ==========================================

os.makedirs("results/nasa", exist_ok=True)

# ==========================================
# Engine 1 Data
# ==========================================

engine = df[df["Engine_ID"] == 1]

# ==========================================
# Sensors to Plot
# ==========================================

sensors = [
    "Sensor2",
    "Sensor3",
    "Sensor7",
    "Sensor11",
    "Sensor15"
]

# ==========================================
# Generate Graphs
# ==========================================

for sensor in sensors:

    plt.figure(figsize=(10,5))

    plt.plot(
        engine["Cycle"],
        engine[sensor],
        linewidth=2
    )

    plt.title(f"{sensor} vs Cycle (Engine 1)")
    plt.xlabel("Cycle")
    plt.ylabel(sensor)

    plt.grid(True)

    save_path = f"results/nasa/{sensor.lower()}_cycle.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {save_path}")

print("\nAll sensor graphs generated successfully!")