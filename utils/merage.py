from pathlib import Path
from rosbags.highlevel import AnyReader
import csv
import numpy as np
from tqdm import tqdm

DATA_DIR = Path("../raw/raw/")
bags = list(DATA_DIR.glob("*.bag"))

TARGET_HZ = 20
SAMPLE_INTERVAL_NS = int(1e9 / TARGET_HZ)  # 50,000,000 ns

# -------------------------
# SAFE MATH
# -------------------------
def safe_diff(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a - b


def safe_float(v):
    try:
        return float(v)
    except:
        return np.nan


output_file = "uav_fault_dataset.csv"
file = open(output_file, "w", newline="")
writer = None

# =========================
# PROCESS EACH BAG
# =========================
for bag in tqdm(bags, desc="Processing bags"):

    with AnyReader([bag]) as reader:

        state = {
            "airspeed_cmd": np.nan, "airspeed_meas": np.nan,
            "roll_cmd": np.nan, "roll_meas": np.nan,
            "pitch_cmd": np.nan, "pitch_meas": np.nan,
            "yaw_cmd": np.nan, "yaw_meas": np.nan,

            "vel_x": np.nan, "vel_y": np.nan, "vel_z": np.nan,

            "imu_roll": np.nan, "imu_pitch": np.nan, "imu_yaw": np.nan,

            "battery_voltage": np.nan,
            "battery_current": np.nan,
            "battery_percent": np.nan,

            "label_engine": 0,
            "label_aileron": 0,
            "label_elevator": 0,
            "label_rudder": 0,
        }

        start_time = None
        next_sample = 0

        for conn, t, raw in reader.messages():

            if start_time is None:
                start_time = t
                next_sample = 0

            msg = reader.deserialize(raw, conn.msgtype)
            topic = conn.topic

            # -------------------------
            # LABELS
            # -------------------------
            if topic.startswith("/failure_status/"):
                val = safe_float(getattr(msg, "data", 0))

                if "engine" in topic:
                    state["label_engine"] = int(val)
                elif "aileron" in topic:
                    state["label_aileron"] = int(val)
                elif "elevator" in topic:
                    state["label_elevator"] = int(val)
                elif "rudder" in topic:
                    state["label_rudder"] = int(val)

            # -------------------------
            # AIRSPEED (cmd vs meas)
            # -------------------------
            elif topic == "/mavros/nav_info/airspeed":
                state["airspeed_cmd"] = safe_float(getattr(msg, "commanded", np.nan))
                state["airspeed_meas"] = safe_float(getattr(msg, "measured", np.nan))

            # -------------------------
            # ROLL
            # -------------------------
            elif topic == "/mavros/nav_info/roll":
                state["roll_cmd"] = safe_float(getattr(msg, "commanded", np.nan))
                state["roll_meas"] = safe_float(getattr(msg, "measured", np.nan))

            # -------------------------
            # PITCH
            # -------------------------
            elif topic == "/mavros/nav_info/pitch":
                state["pitch_cmd"] = safe_float(getattr(msg, "commanded", np.nan))
                state["pitch_meas"] = safe_float(getattr(msg, "measured", np.nan))

            # -------------------------
            # YAW
            # -------------------------
            elif topic == "/mavros/nav_info/yaw":
                state["yaw_cmd"] = safe_float(getattr(msg, "commanded", np.nan))
                state["yaw_meas"] = safe_float(getattr(msg, "measured", np.nan))

            # -------------------------
            # VELOCITY
            # -------------------------
            elif topic == "/mavros/local_position/velocity":
                if hasattr(msg, "twist"):
                    state["vel_x"] = msg.twist.linear.x
                    state["vel_y"] = msg.twist.linear.y
                    state["vel_z"] = msg.twist.linear.z

            # -------------------------
            # IMU
            # -------------------------
            elif topic == "/mavros/imu/data":
                if hasattr(msg, "orientation"):
                    state["imu_roll"] = msg.orientation.x
                    state["imu_pitch"] = msg.orientation.y
                    state["imu_yaw"] = msg.orientation.z

            # -------------------------
            # BATTERY
            # -------------------------
            elif topic == "/mavros/battery":
                state["battery_voltage"] = safe_float(getattr(msg, "voltage", np.nan))
                state["battery_current"] = safe_float(getattr(msg, "current", np.nan))
                state["battery_percent"] = safe_float(getattr(msg, "percentage", np.nan))

            # -------------------------
            # SAMPLING (20Hz)
            # -------------------------
            rel_t = t - start_time

            while rel_t >= next_sample:

                row = {
                    "bag": bag.name,
                    "timestamp_ns": next_sample,

                    # AIRSPEED
                    "airspeed_cmd": state["airspeed_cmd"],
                    "airspeed_meas": state["airspeed_meas"],
                    "airspeed_error": safe_diff(state["airspeed_cmd"], state["airspeed_meas"]),

                    # ROLL
                    "roll_cmd": state["roll_cmd"],
                    "roll_meas": state["roll_meas"],
                    "roll_error": safe_diff(state["roll_cmd"], state["roll_meas"]),

                    # PITCH
                    "pitch_cmd": state["pitch_cmd"],
                    "pitch_meas": state["pitch_meas"],
                    "pitch_error": safe_diff(state["pitch_cmd"], state["pitch_meas"]),

                    # YAW
                    "yaw_cmd": state["yaw_cmd"],
                    "yaw_meas": state["yaw_meas"],
                    "yaw_error": safe_diff(state["yaw_cmd"], state["yaw_meas"]),

                    # MOTION
                    "vel_x": state["vel_x"],
                    "vel_y": state["vel_y"],
                    "vel_z": state["vel_z"],

                    # IMU
                    "imu_roll": state["imu_roll"],
                    "imu_pitch": state["imu_pitch"],
                    "imu_yaw": state["imu_yaw"],

                    # ENERGY
                    "battery_voltage": state["battery_voltage"],
                    "battery_current": state["battery_current"],
                    "battery_percent": state["battery_percent"],

                    # LABELS
                    "engine_fault": state["label_engine"],
                    "aileron_fault": state["label_aileron"],
                    "elevator_fault": state["label_elevator"],
                    "rudder_fault": state["label_rudder"],
                }

                # init writer safely
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=row.keys())
                    writer.writeheader()

                writer.writerow(row)

                next_sample += SAMPLE_INTERVAL_NS

file.close()

print(f"\nDONE → {output_file} created successfully")