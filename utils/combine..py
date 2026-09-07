import pandas as pd

# ============================================================
# Combine UAV train and test datasets for descriptive analysis
# Time-series safe: preserve bag and timestamp ordering
# ============================================================

# Input files
TRAIN_FILE = "../data/train_raw.csv"
TEST_FILE = "../data/test_raw.csv"

# Output file
OUTPUT_FILE = "uav_timeseries.csv"


# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------
print("Loading datasets...")

df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

print(f"Train shape: {df_train.shape}")
print(f"Test shape : {df_test.shape}")


# ------------------------------------------------------------
# 2. Add dataset identifier
# ------------------------------------------------------------
# This allows us to distinguish the original train/test data
# after concatenation.

df_train = df_train.copy()
df_test = df_test.copy()

df_train["dataset"] = "train"
df_test["dataset"] = "test"


# ------------------------------------------------------------
# 3. Combine datasets
# ------------------------------------------------------------
df_all = pd.concat(
    [df_train, df_test],
    axis=0,
    ignore_index=True
)


# ------------------------------------------------------------
# 4. Sort chronologically
# ------------------------------------------------------------
# IMPORTANT:
# Each "bag" represents a separate flight/log.
#
# We must NOT sort only by timestamp_ns because timestamps
# from different bags may overlap.
#
# Therefore:
#     bag → timestamp_ns
#
# This preserves the time-series structure within each flight.

df_all = df_all.sort_values(
    ["bag", "timestamp_ns"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 5. Save combined dataset
# ------------------------------------------------------------
df_all.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 6. Verification
# ------------------------------------------------------------
print("\nCombined dataset saved successfully!")
print(f"Output file: {OUTPUT_FILE}")
print(f"Combined shape: {df_all.shape}")

print("\nColumns:")
print(df_all.columns.tolist())

print("\nFirst rows:")
print(df_all.head())

print("\nDataset distribution:")
print(df_all["dataset"].value_counts())

print("\nNumber of bags:")
print(df_all["bag"].nunique())