from pathlib import Path
from rosbags.highlevel import AnyReader

DATA_DIR = Path(r"C:\Project\data_mining\raw/raw/")
dead_bag = DATA_DIR / "carbonZ_2018-07-18-12-10-11.bag"

print(f"Scanning the entire timeline of {dead_bag.name} for HUD airspeed...")

max_hud_airspeed = 0.0
msg_count = 0

with AnyReader([dead_bag]) as reader:
    for conn, t, raw in reader.messages():
        if conn.topic == "/mavros/vfr_hud":
            msg = reader.deserialize(raw, conn.msgtype)
            msg_count += 1
            
            # Read the airspeed field inside the HUD message structure
            current_airspeed = getattr(msg, "airspeed", 0.0)
            if current_airspeed > max_hud_airspeed:
                max_hud_airspeed = current_airspeed

print("\n=== HUD TOPIC SCAN RESULTS ===")
print(f"Total VFR_HUD messages checked: {msg_count:,}")
print(f"Highest Airspeed found in HUD: {max_hud_airspeed} m/s")
