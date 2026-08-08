import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# File Paths
# -------------------------------------------------

DATA_PATH = "data/ai4i2020.csv"

OUTPUT_DIR = "data"

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# Shuffle Dataset
# -------------------------------------------------

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# -------------------------------------------------
# Split into Three Equal Parts
# -------------------------------------------------

n = len(df)

part = n // 3

factory_A = df.iloc[:part]

factory_B = df.iloc[part:2 * part]

factory_C = df.iloc[2 * part:]

# -------------------------------------------------
# Save Files
# -------------------------------------------------

factory_A.to_csv(
    os.path.join(OUTPUT_DIR, "factory_A.csv"),
    index=False
)

factory_B.to_csv(
    os.path.join(OUTPUT_DIR, "factory_B.csv"),
    index=False
)

factory_C.to_csv(
    os.path.join(OUTPUT_DIR, "factory_C.csv"),
    index=False
)

# -------------------------------------------------
# Summary
# -------------------------------------------------

print("=" * 50)
print("Dataset Split Successfully")
print("=" * 50)

print(f"Factory A : {len(factory_A)} samples")
print(f"Factory B : {len(factory_B)} samples")
print(f"Factory C : {len(factory_C)} samples")

print("=" * 50)