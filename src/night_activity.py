from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("mobile_phone_use/data")
OUT = Path("data/processed")

ok = set(pd.read_csv(OUT / "participants_ok.csv")["pid"])
win = pd.read_csv(OUT / "sleep_windows.csv")[["pid", "n_nights", "bed_min", "wake_min", "night_cover"]]
win["reliable"] = (win["night_cover"] >= 0.5) & (win["n_nights"] >= 7)
ref_bed = win.loc[win["reliable"], "bed_min"].median()
ref_wake = win.loc[win["reliable"], "wake_min"].median()
win["bed_min"] = np.where(win["reliable"], win["bed_min"], ref_bed)
win["wake_min"] = np.where(win["reliable"], win["wake_min"], ref_wake)
win["bed_c"] = (win["bed_min"] + 1080) % 1440    # gün içi dakika
win["wake_c"] = (win["wake_min"] + 1080) % 1440

rng = np.random.default_rng(0)
unrel = list(win.loc[~win["reliable"], "pid"])
rel = list(rng.choice(win.loc[win["reliable"], "pid"], size=80, replace=False))
use = set(unrel) | set(rel)

ev = []
for i, f in enumerate(sorted(DATA.glob("*.csv"))):
    pid = f"P{i:03d}"
    if pid not in ok or pid not in use:
        continue
    df = pd.read_csv(f, usecols=["ts", "sensor_id", "App_Value"], low_memory=False)
    a = df[df["sensor_id"] == "App"].dropna(subset=["App_Value"])
    a = a[~a["App_Value"].str.startswith(("+", "=", "-")) & (a["App_Value"] != "org.pielot.sns2")]
    t = pd.to_datetime(a["ts"])
    ev.append(pd.DataFrame({"pid": pid, "mod": (t.dt.hour * 60 + t.dt.minute).values}))
ev = pd.concat(ev).merge(win[["pid", "bed_c", "wake_c", "reliable"]], on="pid")
wrap = ev["bed_c"] > ev["wake_c"]
ev["night"] = np.where(wrap, (ev["mod"] >= ev["bed_c"]) | (ev["mod"] < ev["wake_c"]),
                       (ev["mod"] >= ev["bed_c"]) & (ev["mod"] < ev["wake_c"]))
cnt = ev.groupby("pid").agg(n_app=("night", "size"), n_night=("night", "sum"))

# ekran saatleri
s = pd.read_parquet(OUT / "sessions.parquet")
s = s[s["valid"] & s["pid"].isin(cnt.index)]
total_h = s.groupby("pid")["duration_s"].sum() / 3600
ov = pd.read_parquet(OUT / "sleep_overlap.parquet")
night_h = ov[ov["pid"].isin(cnt.index)].groupby("pid")["overlap_min"].sum() / 60

r = cnt.join(total_h.rename("total_h")).join(night_h.rename("night_h"))
r = r.join(win.set_index("pid")["reliable"])
r["night_rate"] = r["n_night"] / r["night_h"].where(r["night_h"] > 0.5)
r["day_rate"] = (r["n_app"] - r["n_night"]) / (r["total_h"] - r["night_h"])
r["ratio"] = r["night_rate"] / r["day_rate"]

pd.set_option("display.width", 200)
for flag, g in r.groupby("reliable"):
    print(f"\nGüvenilir pencere = {flag}: {len(g)} katılımcı")
    print("  gece/gündüz olay oranı (ortanca):", round(g["ratio"].median(), 2))
    print("  oranı < 0.30 olan katılımcı payı:", round((g["ratio"] < 0.30).mean(), 2))
    print("  uyku penceresinde ekran saati (ortanca, toplam):", round(g["night_h"].median(), 1))

r["night_suspect"] = (~r["reliable"].astype(bool)) & (r["ratio"] < 0.30)
r[["reliable", "night_h", "ratio", "night_suspect"]].to_csv(OUT / "night_activity.csv")
print("\nŞüpheli (güvenilmez pencere ve oran < 0.30):", int(r["night_suspect"].sum()))