import pandas as pd

# Load your generated master dataset
df = pd.read_csv("../data/attempt_1.csv")

print("=== 1. DATASET VOLUME OVERVIEW ===")
print(f"Total entries ready for Mining: {df.shape[0]:,}")
print(f"Total features extracted: {df.shape[1]}")

print("\n=== 2. CONFIRMING ACTIVE FAILURE TIMESTEPS ===")
# This verifies that your latching failure logic wrote rows correctly
print("Total rows where a fault is actively happening:")
print(f"- Aileron Faults: {df['aileron_fault'].sum():,}")
print(f"- Elevator Faults: {df['elevator_fault'].sum():,}")
print(f"- Rudder Faults: {df['rudder_fault'].sum():,}")
print(f"- Engine Faults: {df['engine_fault'].sum():,}")

print("\n=== 3. TESTING VALUE OF THE FEATURE ENGINEERING ===")
# If your logic worked, the mean error during a fault (1) should be huge 
# compared to a normal healthy flight (0).
if 'roll_error' in df.columns:
    print("\nAverage Roll Error by Aileron Status:")
    print(df.groupby('aileron_fault')['roll_error'].mean())

if 'pitch_error' in df.columns:
    print("\nAverage Pitch Error by Elevator Status:")
    print(df.groupby('elevator_fault')['pitch_error'].mean())
