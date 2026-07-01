from rosbags.highlevel import AnyReader
from pathlib import Path
import pandas as pd
from tqdm import tqdm

bag_folder = Path("../raw/raw/") 
bag_files = list(bag_folder.rglob("*.bag"))

rows = []

print(f"Found {len(bag_files)} bags")

for bag_path in tqdm(bag_files):

    flight_id = bag_path.stem

    # default values
    airspeed = roll = pitch = yaw = None
    battery = velocity = None
    label = 0

    try:
        with AnyReader([bag_path]) as reader:

            for conn, t, raw in reader.messages():

                topic = conn.topic
                msg = reader.deserialize(raw, conn.msgtype)

                # --- FEATURES ---
                if topic == "/mavros/nav_info/airspeed":
                    airspeed = getattr(msg, "data", None)

                elif topic == "/mavros/nav_info/roll":
                    roll = getattr(msg, "data", None)

                elif topic == "/mavros/nav_info/pitch":
                    pitch = getattr(msg, "data", None)

                elif topic == "/mavros/nav_info/yaw":
                    yaw = getattr(msg, "data", None)

                elif topic == "/mavros/battery":
                    battery = getattr(msg, "voltage", None)

                elif topic == "/mavros/local_position/velocity":
                    if hasattr(msg, "twist"):
                        velocity = msg.twist.linear.x

                # --- LABEL ---
                elif topic == "/failure_status/engines":
                    label = int(getattr(msg, "data", 0))

            # one row per bag (IMPORTANT simplification)
            rows.append([
                flight_id,
                airspeed,
                roll,
                pitch,
                yaw,
                battery,
                velocity,
                label
            ])

    except Exception as e:
        print(f"Error in {bag_path}: {e}")

df = pd.DataFrame(rows, columns=[
    "flight_id",
    "airspeed",
    "roll",
    "pitch",
    "yaw",
    "battery",
    "velocity",
    "label"
])

df.to_csv("uav_dataset.csv", index=False)

print("DONE → uav_dataset.csv created")
print(df.head())