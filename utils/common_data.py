from pathlib import Path
from rosbags.highlevel import AnyReader
import pandas as pd

bag_folder = Path(r"../raw/raw/")

rows = []

for bag in bag_folder.glob("*.bag"):

    with AnyReader([bag]) as reader:

        for conn in reader.connections:

            rows.append({
                "bag": bag.name,
                "topic": conn.topic,
                "msgtype": conn.msgtype
            })

df = pd.DataFrame(rows)

summary = (
    df.groupby("topic")["msgtype"]
      .unique()
      .reset_index()
)

print(summary)