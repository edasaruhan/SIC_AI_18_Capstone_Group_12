from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed")
GROUPS = ["kumar", "oyun", "sosyal_video", "genel_web", "dusuk"]

ev = pd.read_parquet(OUT / "app_events.parquet")
ev = ev[ev["exclude"] == 0].copy()
ok = pd.read_csv(OUT / "participants_ok.csv")
ev = ev[ev["pid"].isin(ok["pid"])].copy()
ev["end"] = ev["ts"] + pd.to_timedelta(ev["dwell_s"], unit="s")

win = pd.read_csv(OUT / "sleep_windows.csv")[["pid", "n_nights", "bed_min", "wake_min", "night_cover"]]
win["reliable"] = (win["night_cover"] >= 0.5) & (win["n_nights"] >= 7)
ref_bed = win.loc[win["reliable"], "bed_min"].median()
ref_wake = win.loc[win["reliable"], "wake_min"].median()
win["bed_min"] = np.where(win["reliable"], win["bed_min"], ref_bed)
win["wake_min"] = np.where(win["reliable"], win["wake_min"], ref_wake)
ev = ev.merge(win[["pid", "bed_min", "wake_min"]], on="pid")
D = ev["ts"].dt.normalize()


def overlap_s(day_shift):
    base = D + pd.Timedelta(days=day_shift) + pd.Timedelta(hours=18)
    ws = base + pd.to_timedelta(ev["bed_min"], unit="m")
    we = base + pd.to_timedelta(ev["wake_min"], unit="m")
    lo = np.maximum(ev["ts"].values, ws.values)
    hi = np.minimum(ev["end"].values, we.values)
    return np.clip((hi - lo) / np.timedelta64(1, "s"), 0, None)


ev["date"] = D
ev["dwell_h"] = ev["dwell_s"] / 3600
ev["sleep_h"] = (overlap_s(-1) + overlap_s(0)) / 3600
ev["is_short"] = ev["dwell_s"] < 10

tot = ev.groupby(["pid", "date"]).agg(n_events=("dwell_h", "size"), app_h=("dwell_h", "sum"),
                                      app_short_share=("is_short", "mean"),
                                      sleep_use_h=("sleep_h", "sum"))
tot["as2"] = np.where(tot["app_h"] > 0, tot["n_events"] / (tot["app_h"] * 60), np.nan)


def pivot(values, suffix):
    p = ev.pivot_table(index=["pid", "date"], columns="risk_group", values=values,
                       aggfunc="sum", fill_value=0)
    return p.reindex(columns=GROUPS, fill_value=0).add_suffix(suffix)


apps = tot.join(pivot("dwell_h", "_h")).join(pivot("sleep_h", "_sleep_h")).reset_index()

daily = pd.read_parquet(OUT / "daily_features.parquet").rename(columns={"short_share": "sess_short_share"})
master = daily.merge(apps, on=["pid", "date"], how="left")
fill0 = ["n_events", "app_h", "sleep_use_h"] + [f"{g}_h" for g in GROUPS] + [f"{g}_sleep_h" for g in GROUPS]
master[fill0] = master[fill0].fillna(0)

na = pd.read_csv(OUT / "night_activity.csv")
susp = set(na.loc[na["night_suspect"], "pid"])
master["night_suspect"] = master["pid"].isin(susp)
master = master.merge(win[["pid", "reliable"]].rename(columns={"reliable": "sleep_window_reliable"}),
                      on="pid", how="left")
master.to_parquet(OUT / "master_daily.parquet")

pd.set_option("display.width", 200)
m = master[~master["night_suspect"]]
print("Katılımcı-gün:", len(master), "| şüpheli hariç:", len(m), "| katılımcı:", m["pid"].nunique())
print("\nGünlük ortalama saat, risk grubuna göre (şüpheli hariç):")
print(m[[f"{g}_h" for g in GROUPS]].mean().round(3))
print("\nUyku penceresindeki ortalama saat/gün:")
print(m[[f"{g}_sleep_h" for g in GROUPS]].mean().round(3))
print("\nOyun oynanan gün payı:", round((m["oyun_h"] > 0).mean(), 3),
      "| uyku penceresinde oyun oynanan gün payı:", round((m["oyun_sleep_h"] > 0).mean(), 3))
print("\nAS2 (uygulama değişimi/dk) özeti:")
print(m["as2"].describe().round(2))