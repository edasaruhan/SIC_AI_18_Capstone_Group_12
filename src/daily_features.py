from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed")
s = pd.read_parquet(OUT / "sessions.parquet")
ok = pd.read_csv(OUT / "participants_ok.csv")
s = s[s["valid"] & s["pid"].isin(ok["pid"])].copy()
s["end_local"] = s["start"] + pd.to_timedelta(s["duration_s"], unit="s")

# 1) Gece yarısını aşan oturumları gün sınırında böl
cross = s["start"].dt.normalize() != s["end_local"].dt.normalize()
rows = []
for r in s[cross].itertuples():
    cur = r.start
    while cur < r.end_local:
        nxt = min(cur.normalize() + pd.Timedelta(days=1), r.end_local)
        rows.append((r.pid, cur, nxt, (nxt - cur).total_seconds()))
        cur = nxt
parts = pd.DataFrame(rows, columns=["pid", "start", "end_local", "duration_s"])
base = s.loc[~cross, ["pid", "start", "end_local", "duration_s"]]
seg = pd.concat([base, parts], ignore_index=True)
seg["date"] = seg["start"].dt.normalize()

# 2) Segment başına gece (00:00-06:00) saniyesi
st, en, d0 = seg["start"].values, seg["end_local"].values, seg["date"].values
night_end = d0 + np.timedelta64(6, "h")
ov = (np.minimum(en, night_end) - np.maximum(st, d0)) / np.timedelta64(1, "s")
seg["night_s"] = np.clip(ov, 0, None)

g = seg.groupby(["pid", "date"]).agg(screen_s=("duration_s", "sum"),
                                     night_s=("night_s", "sum"))
g["screen_h"] = g["screen_s"] / 3600
g["night_ratio"] = g["night_s"] / g["screen_s"]

# 3) Oturum bazlı özellikler (orijinal oturumlar, başlangıç gününe göre)
s["date"] = s["start"].dt.normalize()
h = s.groupby(["pid", "date"]).agg(
    n_sessions=("duration_s", "size"),
    short_share=("duration_s", lambda x: (x < 30).mean()),
    n_unlocked=("unlocked", "sum"),
)

daily = g[["screen_h", "night_ratio"]].join(h, how="outer").reset_index()
daily = daily.merge(ok[["pid", "unlock_reliable"]], on="pid")
daily.loc[~daily["unlock_reliable"], "n_unlocked"] = np.nan
daily.to_parquet(OUT / "daily_features.parquet")

pd.set_option("display.width", 200)
print("Katılımcı:", daily["pid"].nunique(), "| katılımcı-gün:", len(daily))
print("24 saati aşan gün sayısı:", int((daily["screen_h"] > 24).sum()))
print(daily[["screen_h", "night_ratio", "n_sessions", "short_share",
             "n_unlocked"]].describe().round(2))