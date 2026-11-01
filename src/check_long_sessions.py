from pathlib import Path
import pandas as pd
from sessions import load_participant, sessionize

files = sorted(Path("mobile_phone_use/data").glob("*.csv"))
df = load_participant(files[0])
s = sessionize(df)

long_ = s[s.duration_s > 7200].copy()


def count_events(row, sensor):
    m = (df.ts >= row.start) & (df.ts <= row.end) & (df.sensor_id == sensor)
    return int(m.sum())


long_["app_events"] = long_.apply(count_events, axis=1, sensor="App")
long_["useract_events"] = long_.apply(count_events, axis=1, sensor="UserAct")
long_["minutes"] = (long_.duration_s / 60).round()

chg = (df[df.sensor_id == "Charging"][["ts", "Charging_Value"]]
       .sort_values("ts").rename(columns={"ts": "chg_ts"}))
long_ = pd.merge_asof(long_.sort_values("start"), chg,
                      left_on="start", right_on="chg_ts", direction="backward")

print(long_[["hour", "minutes", "app_events", "useract_events",
             "Charging_Value", "unlocked"]].to_string(index=False))