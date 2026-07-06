from pathlib import Path
from rosbags.highlevel import AnyReader

bag = Path("../raw/raw/carbonZ_2018-07-18-12-10-11.bag")

with AnyReader([bag]) as reader:
    for conn, t, raw in reader.messages():
        print(type(t))
        print(t)
        break
    
first = None
last = None

with AnyReader([bag]) as reader:
    for conn, t, raw in reader.messages():
        if first is None:
            first = t
        last = t

print("First:", first)
print("Last :", last)
print("Duration:", last - first)

count = 0

with AnyReader([bag]) as reader:
    for _ in reader.messages():
        count += 1

print("Messages:", count)


with AnyReader([bag]) as reader:
    for conn, t, raw in reader.messages():
        if conn.topic == "/mavros/battery":
            msg = reader.deserialize(raw, conn.msgtype)
            print(msg)
            break