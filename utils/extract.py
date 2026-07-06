from pathlib import Path
from rosbags.highlevel import AnyReader
import csv
import numpy as np
from tqdm import tqdm

# CONFIG
DATA_DIR = Path("../raw/raw/")
bags = sorted(list(DATA_DIR.glob("*.bag")))
OUTPUT_FILE = "attempt_1.csv"

# SAFE FUNCTIONS
def safe_float(v):
    try:
        return float(v)
    except:
        return np.nan


def safe_diff(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a - b


# WRITE SETUP
file = open(OUTPUT_FILE, "w", newline="")
writer = None


# PROCESS ALL BAGS
for bag in tqdm(bags, desc="Processing bags"):

    with AnyReader([bag]) as reader:


        # GLOBAL STATE MEMORY
        state = {
            # airspeed
            "airspeed_cmd": np.nan,
            "airspeed_meas": np.nan,

            # attitude
            "roll_cmd": np.nan,
            "roll_meas": np.nan,
            "pitch_cmd": np.nan,
            "pitch_meas": np.nan,
            "yaw_cmd": np.nan,
            "yaw_meas": np.nan,

            # velocity
            "vel_x": np.nan,
            "vel_y": np.nan,
            "vel_z": np.nan,

            # imu (NOTE: quaternion raw)
            "imu_x": np.nan,
            "imu_y": np.nan,
            "imu_z": np.nan,

            # battery
            "battery_voltage": np.nan,
            "battery_current": np.nan,
            "battery_percent": np.nan,

            # labels
            "engine_fault": 0,
            "aileron_fault": 0,
            "elevator_fault": 0,
            "rudder_fault": 0,
        }

        # STREAM PROCESSING
        for conn, t, raw in reader.messages():

            msg = reader.deserialize(raw, conn.msgtype)
            topic = conn.topic

            # FAILURE LABELS
            if topic.startswith("/failure_status/"):
                val = safe_float(getattr(msg, "data", 0))

                if "engine" in topic:
                    state["engine_fault"] = int(val)
                elif "aileron" in topic:
                    state["aileron_fault"] = int(val)
                elif "elevator" in topic:
                    state["elevator_fault"] = int(val)
                elif "rudder" in topic:
                    state["rudder_fault"] = int(val)

            # AIRSPEED (REFERENCE TRIGGER)
            elif topic == "/mavros/nav_info/airspeed":

                state["airspeed_cmd"] = safe_float(getattr(msg, "commanded", np.nan))
                state["airspeed_meas"] = safe_float(getattr(msg, "measured", np.nan))

                # IMPORTANT: write ONE row per airspeed message
                row = {
                    "bag": bag.name,
                    "timestamp_ns": t,

                    # AIRSPEED
                    "airspeed_cmd": state["airspeed_cmd"],
                    "airspeed_meas": state["airspeed_meas"],
                    "airspeed_error": safe_diff(
                        state["airspeed_cmd"],
                        state["airspeed_meas"]
                    ),

                    # ATTITUDE
                    "roll_cmd": state["roll_cmd"],
                    "roll_meas": state["roll_meas"],
                    "roll_error": safe_diff(state["roll_cmd"], state["roll_meas"]),

                    "pitch_cmd": state["pitch_cmd"],
                    "pitch_meas": state["pitch_meas"],
                    "pitch_error": safe_diff(state["pitch_cmd"], state["pitch_meas"]),

                    "yaw_cmd": state["yaw_cmd"],
                    "yaw_meas": state["yaw_meas"],
                    "yaw_error": safe_diff(state["yaw_cmd"], state["yaw_meas"]),

                    # VELOCITY
                    "vel_x": state["vel_x"],
                    "vel_y": state["vel_y"],
                    "vel_z": state["vel_z"],

                    # IMU (RAW QUATERNION)
                    "imu_x": state["imu_x"],
                    "imu_y": state["imu_y"],
                    "imu_z": state["imu_z"],

                    # BATTERY
                    "battery_voltage": state["battery_voltage"],
                    "battery_current": state["battery_current"],
                    "battery_percent": state["battery_percent"],

                    # LABELS
                    "engine_fault": state["engine_fault"],
                    "aileron_fault": state["aileron_fault"],
                    "elevator_fault": state["elevator_fault"],
                    "rudder_fault": state["rudder_fault"],
                }

                # INIT CSV WRITER
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=row.keys())
                    writer.writeheader()

                writer.writerow(row)

            # ROLL
            elif topic == "/mavros/nav_info/roll":
                state["roll_cmd"] = safe_float(getattr(msg, "commanded", np.nan))
                state["roll_meas"] = safe_float(getattr(msg, "measured", np.nan))

            # PITCH
            elif topic == "/mavros/nav_info/pitch":
                state["pitch_cmd"] = safe_float(getattr(msg, "commanded", np.nan))
                state["pitch_meas"] = safe_float(getattr(msg, "measured", np.nan))

            # YAW
            elif topic == "/mavros/nav_info/yaw":
                state["yaw_cmd"] = safe_float(getattr(msg, "commanded", np.nan))
                state["yaw_meas"] = safe_float(getattr(msg, "measured", np.nan))

            # VELOCITY
            elif topic == "/mavros/local_position/velocity":
                if hasattr(msg, "twist"):
                    state["vel_x"] = msg.twist.linear.x
                    state["vel_y"] = msg.twist.linear.y
                    state["vel_z"] = msg.twist.linear.z

            # IMU
            elif topic == "/mavros/imu/data":
                if hasattr(msg, "orientation"):
                    state["imu_x"] = msg.orientation.x
                    state["imu_y"] = msg.orientation.y
                    state["imu_z"] = msg.orientation.z

            # BATTERY
            elif topic == "/mavros/battery":
                state["battery_voltage"] = safe_float(getattr(msg, "voltage", np.nan))
                state["battery_current"] = safe_float(getattr(msg, "current", np.nan))
                state["battery_percent"] = safe_float(getattr(msg, "percentage", np.nan))


file.close()

print(f"\nDONE → {OUTPUT_FILE} created successfully")