from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed")
s = pd.read_parquet(OUT / "sessions.parquet")
ok = pd.read_csv(OUT / "participants_ok.csv")
s = s[s["valid"] & s["pid"].isin(ok["pid"])].copy()
s["end_local"] = s["start"] + pd.to_timedelta(s["duration_s"], unit="s")

win = pd.read_csv(OUT / "sleep_windows.csv")[["pid", "n_nights", "bed_min", "wake_min", "night_cover"]]
win["reliable"] = (win["night_cover"] >= 0.5) & (win["n_nights"] >= 7)
ref_bed = win.loc[win["reliable"], "bed_min"].median()
ref_wake = win.loc[win["reliable"], "wake_min"].median()
win["bed_min"] = np.where(win["reliable"], win["bed_min"], ref_bed)
win["wake_min"] = np.where(win["reliable"], win["wake_min"], ref_wake)

s = s.merge(win[["pid", "bed_min", "wake_min", "reliable"]], on="pid")
D = s["start"].dt.normalize()


def overlap_s(day_shift):
    # night = (oturum günü + day_shift); pencere o günün 18:00'inden itibaren ölçülür
    base = D + pd.Timedelta(days=day_shift) + pd.Timedelta(hours=18)
    ws = base + pd.to_timedelta(s["bed_min"], unit="m")
    we = base + pd.to_timedelta(s["wake_min"], unit="m")
    lo = np.maximum(s["start"].values, ws.values)
    hi = np.minimum(s["end_local"].values, we.values)
    return np.clip((hi - lo) / np.timedelta64(1, "s"), 0, None)


prev = pd.DataFrame({"pid": s["pid"], "night": D - pd.Timedelta(days=1),
                     "ov": overlap_s(-1), "reliable": s["reliable"]})
same = pd.DataFrame({"pid": s["pid"], "night": D,
                     "ov": overlap_s(0), "reliable": s["reliable"]})
n = pd.concat([prev, same]).groupby(["pid", "night", "reliable"], as_index=False)["ov"].sum()
n["overlap_min"] = n["ov"] / 60
n.drop(columns="ov").to_parquet(OUT / "sleep_overlap.parquet")

print("Güvenilir pencereli katılımcı:", int(win["reliable"].sum()), "/", len(win))
print("Yedek pencere (ortanca yatış/kalkış):", ref_bed, ref_wake, "(18:00'den itibaren dakika)")
print("Gece sayısı:", len(n))
for flag, g in n.groupby("reliable"):
    pos = g[g["overlap_min"] >= 1]
    print(f"\nGüvenilir pencere = {flag}: {len(g)} gece")
    print("  >=1 dk çakışma payı:", round((g["overlap_min"] >= 1).mean(), 2))
    print("  >=30 dk çakışma payı:", round((g["overlap_min"] >= 30).mean(), 2))
    print("  çakışanlarda ortanca dk:", round(pos["overlap_min"].median(), 1))