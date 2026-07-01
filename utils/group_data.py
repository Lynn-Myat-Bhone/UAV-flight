import pandas as pd

df = pd.read_csv("bag_topics_inventory.csv")  


def classify_topic(topic):
    topic = topic.lower()


    # LABELS (Ground Truth)
    if topic.startswith("/failure_status/"):
        return "label"

    # FLIGHT STATE
    if any(x in topic for x in [
        "imu",
        "roll",
        "pitch",
        "yaw",
        "airspeed",
        "velocity",
        "local_position",
        "odom",
        "pose",
        "wind_estimation"
    ]):
        return "flight_state"

    # NAVIGATION
    if any(x in topic for x in [
        "gps",
        "global_position",
        "compass",
        "rel_alt",
        "fix",
        "heading",
        "home_position"
    ]):
        return "navigation"

    # ENERGY
    if any(x in topic for x in [
        "battery",
        "voltage",
        "current",
        "power",
        "vfr_hud"
    ]):
        return "energy"

    # CONTROL
    if any(x in topic for x in [
        "rc",
        "mavctrl",
        "setpoint",
        "command"
    ]):
        return "control"

    # SYSTEM / DEBUG
    if any(x in topic for x in [
        "rosout",
        "diagnostic",
        "mavlink",
        "sysid",
        "mission",
        "tf_static",
        "state",
        "time_reference",
        "traj_file"
    ]):
        return "system"

    return "unknown"


# Apply grouping
df["group"] = df["topic"].apply(classify_topic)

# Save grouped result
df.to_csv("grouped_topics.csv", index=False)

print("\n=== Group Counts ===")
print(df["group"].value_counts())

print("\n=== Unique Topics per Group ===")
summary = (
    df.groupby("group")["topic"]
      .nunique()
      .reset_index()
      .rename(columns={"topic": "unique_topics"})
)

print(summary)

# Save unknown topics for inspection
unknown = df[df["group"] == "unknown"]
unknown.to_csv("unknown_topics.csv", index=False)

print("\nUnknown topics saved to unknown_topics.csv")
print("\n=== GROUP SUMMARY ===")

for group in sorted(df["group"].unique()):

    total_rows = len(df[df["group"] == group])

    unique_topics = (
        df[df["group"] == group]["topic"]
        .nunique()
    )

    print(
        f"{group:<15} "
        f"Rows={total_rows:<5} "
        f"UniqueTopics={unique_topics}"
    )
    
flight_topics = sorted(
    df[df["group"] == "flight_state"]["topic"].unique()
)

print("\n===== FLIGHT STATE TOPICS =====")

for t in flight_topics:
    print(t)