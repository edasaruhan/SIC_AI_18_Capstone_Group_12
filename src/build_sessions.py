"""1. adım: ham ekran olaylarından (On / Unlocked / Off) oturum çıkarır.

Çıktı: sessions.parquet, participant_summary.csv, pid_map.csv (kimlik eşlemesi, yerelde kalır)
"""
import pandas as pd
from common import PROCESSED, raw_files

MAX_SESSION_S = 12 * 3600   # daha uzun oturumlar büyük ihtimalle kaçmış bir Off kaydı


def load_screen_events(path):
    df = pd.read_csv(path, usecols=["ts", "ts_raw", "sensor_id", "Screen_Value"], low_memory=False)
    screen = df.sort_values("ts_raw").query("sensor_id == 'Screen'").copy()
    screen["ts"] = pd.to_datetime(screen["ts"])
    return screen


def sessionize(screen):
    """Oturum = ilk On (veya On'u kaçmış Unlocked) ile sonraki Off arası.

    Süre ts_raw'dan (UTC, ms) hesaplanır; yerel saat (ts) yaz saati geçişinde sıçradığı için kullanılmaz.
    """
    sessions = []
    start = start_raw = unlocked_at = None
    for ts, raw, state in zip(screen["ts"], screen["ts_raw"], screen["Screen_Value"]):
        if state == "On" and start is None:
            start, start_raw = ts, raw
        elif state == "Unlocked":
            if start is None:
                start, start_raw = ts, raw
            if unlocked_at is None:
                unlocked_at = ts
        elif state == "Off" and start is not None:
            sessions.append((start, ts, unlocked_at, (raw - start_raw) / 1000))
            start = start_raw = unlocked_at = None
    out = pd.DataFrame(sessions, columns=["start", "end", "unlocked_at", "duration_s"])
    out["unlocked"] = out["unlocked_at"].notna()
    out["date"] = out["start"].dt.date
    out["hour"] = out["start"].dt.hour
    out["valid"] = (out["duration_s"] > 0) & (out["duration_s"] <= MAX_SESSION_S)
    return out


def summarize(pid, sessions):
    valid = sessions[sessions["valid"]]
    n_days = max(valid["date"].nunique(), 1)
    return {"pid": pid, "n_days": n_days, "n_sessions": len(sessions),
            "invalid_share": 1 - sessions["valid"].mean(),
            "unlocked_share": valid["unlocked"].mean(),
            "screen_h_per_day": valid["duration_s"].sum() / 3600 / n_days,
            "median_session_s": valid["duration_s"].median()}


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    all_sessions, summaries, pid_map = [], [], []
    for pid, path in raw_files():
        pid_map.append({"pid": pid, "file_stem": path.stem})
        sessions = sessionize(load_screen_events(path))
        if sessions.empty:
            summaries.append({"pid": pid, "n_sessions": 0})
            continue
        sessions.insert(0, "pid", pid)
        all_sessions.append(sessions)
        summaries.append(summarize(pid, sessions))

    sessions = pd.concat(all_sessions, ignore_index=True)
    summary = pd.DataFrame(summaries)
    sessions.to_parquet(PROCESSED / "sessions.parquet")
    summary.to_csv(PROCESSED / "participant_summary.csv", index=False)
    pd.DataFrame(pid_map).to_csv(PROCESSED / "pid_map.csv", index=False)

    print("Katılımcı:", len(summary), "| oturum:", len(sessions),
          "| geçersiz oturum payı:", round(1 - sessions["valid"].mean(), 4))
    print(summary.drop(columns="pid").describe().round(2))


if __name__ == "__main__":
    main()
