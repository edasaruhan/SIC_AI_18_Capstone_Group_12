"""3. adım: oturumlardan günlük ekran özellikleri.

Çıktı: daily_features.parquet (pid, date, screen_h, night_ratio, n_sessions, short_share, n_unlocked)
"""
import numpy as np
import pandas as pd
from common import PROCESSED, load_ok_participants, load_valid_sessions

NIGHT_END_H = 6          # gece oranı: 00:00-06:00 arasındaki ekran süresinin payı
SHORT_SESSION_S = 30


def split_at_midnight(sessions):
    """Gece yarısını aşan oturumları gün sınırında böler; süre doğru güne yazılsın diye."""
    crosses = sessions["start"].dt.normalize() != sessions["end_local"].dt.normalize()
    pieces = []
    for row in sessions[crosses].itertuples():
        cursor = row.start
        while cursor < row.end_local:
            piece_end = min(cursor.normalize() + pd.Timedelta(days=1), row.end_local)
            pieces.append((row.pid, cursor, piece_end, (piece_end - cursor).total_seconds()))
            cursor = piece_end
    cols = ["pid", "start", "end_local", "duration_s"]
    segments = pd.concat([sessions.loc[~crosses, cols], pd.DataFrame(pieces, columns=cols)], ignore_index=True)
    segments["date"] = segments["start"].dt.normalize()
    return segments


def screen_time_per_day(segments):
    day_start = segments["date"].values
    night_end = day_start + np.timedelta64(NIGHT_END_H, "h")
    night_s = (np.minimum(segments["end_local"].values, night_end)
               - np.maximum(segments["start"].values, day_start)) / np.timedelta64(1, "s")
    segments = segments.assign(night_s=np.clip(night_s, 0, None))
    daily = segments.groupby(["pid", "date"]).agg(screen_s=("duration_s", "sum"), night_s=("night_s", "sum"))
    daily["screen_h"] = daily["screen_s"] / 3600
    daily["night_ratio"] = daily["night_s"] / daily["screen_s"]
    return daily[["screen_h", "night_ratio"]]


def session_counts_per_day(sessions):
    """Oturum sayıları, oturumun başladığı güne yazılır."""
    sessions = sessions.assign(date=sessions["start"].dt.normalize())
    return sessions.groupby(["pid", "date"]).agg(
        n_sessions=("duration_s", "size"),
        short_share=("duration_s", lambda d: (d < SHORT_SESSION_S).mean()),
        n_unlocked=("unlocked", "sum"))


def main():
    sessions = load_valid_sessions()
    daily = (screen_time_per_day(split_at_midnight(sessions))
             .join(session_counts_per_day(sessions), how="outer").reset_index())
    daily = daily.merge(load_ok_participants()[["pid", "unlock_reliable"]], on="pid")
    daily.loc[~daily["unlock_reliable"], "n_unlocked"] = np.nan
    daily.to_parquet(PROCESSED / "daily_features.parquet")

    print("Katılımcı:", daily["pid"].nunique(), "| katılımcı-gün:", len(daily))
    print(daily.drop(columns=["pid", "date", "unlock_reliable"]).describe().round(2))


if __name__ == "__main__":
    main()
