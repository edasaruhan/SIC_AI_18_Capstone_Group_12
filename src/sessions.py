from pathlib import Path
import pandas as pd

COLS = ["uuid", "ts", "ts_raw", "sensor_id", "Screen_Value", "App_Value", "Charging_Value"]
MAX_SESSION_S = 12 * 3600  # bundan uzun oturumlar büyük ihtimalle eksik Off kaydı


def load_participant(path):
    df = pd.read_csv(path, usecols=COLS, low_memory=False)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts_raw").reset_index(drop=True)


def sessionize(df):
    scr = df.loc[df.sensor_id == "Screen", ["ts", "ts_raw", "Screen_Value"]]
    sessions = []
    start = start_raw = unlocked_at = None
    for ts, raw, state in zip(scr["ts"], scr["ts_raw"], scr["Screen_Value"]):
        if state == "On":
            if start is None:
                start, start_raw, unlocked_at = ts, raw, None
        elif state == "Unlocked":
            if start is None:
                start, start_raw, unlocked_at = ts, raw, ts  # On kaydı kaçmış
            elif unlocked_at is None:
                unlocked_at = ts
        elif state == "Off":
            if start is not None:
                sessions.append((start, ts, unlocked_at, (raw - start_raw) / 1000))
                start = start_raw = unlocked_at = None
    out = pd.DataFrame(sessions, columns=["start", "end", "unlocked_at", "duration_s"])
    out["unlocked"] = out["unlocked_at"].notna()
    out["date"] = out["start"].dt.date
    out["hour"] = out["start"].dt.hour
    out["valid"] = (out["duration_s"] > 0) & (out["duration_s"] <= MAX_SESSION_S)
    return out