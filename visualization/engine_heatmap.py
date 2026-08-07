
import pandas as pd
import matplotlib.pyplot as plt
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

# ==========================================
# Engine 1
# ==========================================

engine = df[df["Engine_ID"] == 1]

# Only sensor columns
sensor_data = engine.iloc[:, 5:]

plt.figure(figsize=(14,6))

plt.imshow(
    sensor_data.T,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(label="Sensor Value")

plt.xlabel("Cycle")
plt.ylabel("Sensors")
plt.title("Engine Sensor Heatmap")

plt.yticks(
    range(len(sensor_data.columns)),
    sensor_data.columns,
    fontsize=8
)

os.makedirs("results/nasa", exist_ok=True)

plt.savefig(
    "results/nasa/engine_sensor_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Engine Sensor Heatmap Saved!")