from pathlib import Path
import numpy as np
import pandas as pd
from sessions import load_participant, sessionize

files = sorted(Path("mobile_phone_use/data").glob("*.csv"))
df = load_participant(files[0])
s = sessionize(df)
act = np.sort(df.loc[df.sensor_id == "UserAct", "ts"].values)


def active_seconds(start, end, T):
    lo, hi = np.searchsorted(act, [np.datetime64(start), np.datetime64(end)])
    pts = np.concatenate([[np.datetime64(start)], act[lo:hi], [np.datetime64(end)]])
    gaps = np.diff(pts).astype("timedelta64[s]").astype(float)
    return np.minimum(gaps, T).sum()


days = s["date"].nunique()
print("Ekran açık süre (saat/gün):", round(s.duration_s.sum() / 3600 / days, 2))
for T in (60, 120, 300):
    col = f"active_{T}"
    s[col] = [active_seconds(a, b, T) for a, b in zip(s.start, s.end)]
    print(f"Aktif süre, boşluk sınırı {T} sn (saat/gün):",
          round(s[col].sum() / 3600 / days, 2),
          "| medyan oturum (sn):", round(s[col].median(), 1))