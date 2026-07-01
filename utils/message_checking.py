from pathlib import Path
from rosbags.highlevel import AnyReader

bag = Path("../raw/raw/carbonZ_2018-07-18-15-53-31.bag")

targets = {
    "/mavros/nav_info/airspeed",
    "/mavros/battery",
    "/mavros/imu/data"
}

seen = set()

with AnyReader([bag]) as reader:

    for conn, ts, raw in reader.messages():

        if conn.topic in targets and conn.topic not in seen:

            msg = reader.deserialize(raw, conn.msgtype)

            print("\n==============================")
            print("TOPIC:", conn.topic)
            print("TYPE:", conn.msgtype)
            print("MESSAGE:")
            print(msg)

            seen.add(conn.topic)

        if len(seen) == len(targets):
            break