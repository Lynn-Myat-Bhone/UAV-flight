from pathlib import Path
from rosbags.highlevel import AnyReader

DATA_DIR = Path(r"C:\Project\data_mining\raw/raw/")
# Target one of your confirmed dead files
dead_bag = DATA_DIR / "carbonZ_2018-07-18-12-10-11.bag"

print(f"Scanning topic structures inside dead file: {dead_bag.name}")

with AnyReader([dead_bag]) as reader:
    found_alternatives = []
    for connection in reader.connections:
        # Check for VFR_HUD or other common airspeed message placeholders
        if "vfr_hud" in connection.topic.lower() or "airspeed" in connection.topic.lower():
            found_alternatives.append((connection.topic, connection.msgtype))
            
    if found_alternatives:
        print("\nFOUND ALTERNATIVES! The missing data might be here:")
        for topic, msgtype in found_alternatives:
            print(f"-> Topic: '{topic}' | Type: {msgtype}")
    else:
        print("\nNo alternative airspeed topics exist in this file.")
