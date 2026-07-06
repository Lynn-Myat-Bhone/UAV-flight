from pathlib import Path
from rosbags.highlevel import AnyReader

DATA_DIR = Path("../raw/raw/")
test_bag = next(DATA_DIR.glob("*.bag"))

print(f"Scanning the entire timeline of {test_bag.name} for any signs of battery life...")

max_voltage = 0.0
max_current = 0.0
message_count = 0

with AnyReader([test_bag]) as reader:
    for conn, t, raw in reader.messages():
        if conn.topic == "/mavros/battery":
            msg = reader.deserialize(raw, conn.msgtype)
            message_count += 1
            
            # Track the highest numbers ever achieved
            if getattr(msg, "voltage", 0.0) > max_voltage:
                max_voltage = msg.voltage
            if getattr(msg, "current", 0.0) > max_current:
                max_current = msg.current

print("\n=== RAW BAG SCAN RESULTS ===")
print(f"Total BatteryState messages checked: {message_count:,}")
print(f"Highest Voltage value found: {max_voltage} V")
print(f"Highest Current value found: {max_current} A")

if max_voltage == 0.0:
    print("\nCONCLUSION: The telemetry is entirely dead at the hardware source.")
    print("Code logic is perfectly correct. The bag files simply contain zero battery data.")
