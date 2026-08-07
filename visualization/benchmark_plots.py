import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Paths
# ==========================================

RESULTS_DIR = "results/benchmark"
CSV_FILE = os.path.join(RESULTS_DIR, "results.csv")

# Create results directory if needed
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# Load CSV
# ==========================================

df = pd.read_csv(CSV_FILE)

print(df)