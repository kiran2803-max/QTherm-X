import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET
from config import DATASET

# ==========================================
# Load NASA C-MAPSS Dataset
# ==========================================

file_path = f"datasets/NASA_CMAPSS/train_{DATASET}.txt"

df = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None
)

# ==========================================
# Assign Column Names
# ==========================================

columns = [
    "Engine_ID",
    "Cycle",
    "Setting1",
    "Setting2",
    "Setting3",

    "Sensor1",
    "Sensor2",
    "Sensor3",
    "Sensor4",
    "Sensor5",
    "Sensor6",
    "Sensor7",
    "Sensor8",
    "Sensor9",
    "Sensor10",
    "Sensor11",
    "Sensor12",
    "Sensor13",
    "Sensor14",
    "Sensor15",
    "Sensor16",
    "Sensor17",
    "Sensor18",
    "Sensor19",
    "Sensor20",
    "Sensor21"
]

df.columns = columns
# ==========================================
# Compute Remaining Useful Life (RUL)
# ==========================================

max_cycle = df.groupby("Engine_ID")["Cycle"].max()

df["RUL"] = (
    max_cycle[df["Engine_ID"]].values
    - df["Cycle"]
)

print("\nFirst 10 Rows with RUL:")
print(df[["Engine_ID", "Cycle", "RUL"]].head(10))

print("=" * 50)
print("NASA C-MAPSS Dataset")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())