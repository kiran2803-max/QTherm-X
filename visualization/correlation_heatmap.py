
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
# Correlation Matrix
# ==========================================

corr = df.corr(numeric_only=True)

# ==========================================
# Save Folder
# ==========================================

os.makedirs("results/nasa", exist_ok=True)

# ==========================================
# Plot
# ==========================================

plt.figure(figsize=(14,12))

plt.imshow(corr, aspect="auto")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90,
    fontsize=8
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns,
    fontsize=8
)

plt.title("NASA C-MAPSS Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "results/nasa/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Correlation heatmap saved successfully!")