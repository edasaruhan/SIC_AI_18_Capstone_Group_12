from pathlib import Path
import pandas as pd

OUT = Path("data/processed")
s = pd.read_parquet(OUT / "sessions.parquet")
ok = pd.read_csv(OUT / "participants_ok.csv")
s = s[s["valid"] & s["pid"].isin(ok["pid"])].copy()
s["end_local"] = s["start"] + pd.to_timedelta(s["duration_s"], unit="s")
s = s.sort_values(["pid", "start"])

# ardışık oturumlar arasındaki ekran kapalı boşluklar
s["next_start"] = s.groupby("pid")["start"].shift(-1)
gaps = s.loc[s["next_start"].notna(), ["pid", "end_local", "next_start"]].rename(
    columns={"end_local": "gap_start", "next_start": "gap_end"})
gaps["gap_h"] = (gaps["gap_end"] - gaps["gap_start"]).dt.total_seconds() / 3600

h = gaps["gap_start"].dt.hour
gaps = gaps[(gaps["gap_h"] >= 3) & (gaps["gap_h"] <= 14) & ((h >= 18) | (h < 9))].copy()
gaps["night"] = (gaps["gap_start"] - pd.Timedelta(hours=12)).dt.normalize()

# her gece için en uzun boşluk
best = gaps.sort_values("gap_h").groupby(["pid", "night"]).tail(1).copy()
ref = best["night"] + pd.Timedelta(hours=18)  # o gecenin 18:00'i
best["bed_min"] = (best["gap_start"] - ref).dt.total_seconds() / 60
best["wake_min"] = (best["gap_end"] - ref).dt.total_seconds() / 60


def clock(m):
    if pd.isna(m):
        return None
    m = int(round(m)) + 18 * 60
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


p = best.groupby("pid").agg(n_nights=("gap_h", "size"), bed_min=("bed_min", "median"),
                            wake_min=("wake_min", "median"), sleep_gap_h=("gap_h", "median"))
days = s.groupby("pid")["start"].apply(lambda x: x.dt.normalize().nunique()).rename("n_days")
p = p.join(days, how="right")
p["n_nights"] = p["n_nights"].fillna(0)
p["night_cover"] = p["n_nights"] / p["n_days"]
p["bedtime"] = p["bed_min"].map(clock)
p["waketime"] = p["wake_min"].map(clock)
p.to_csv(OUT / "sleep_windows.csv")

pd.set_option("display.width", 200)
print("Katılımcı:", len(p), "| uyku boşluğu bulunan:", int((p["n_nights"] > 0).sum()))
print("Ortanca yatış:", clock(p["bed_min"].median()), "| ortanca kalkış:", clock(p["wake_min"].median()))
print("\nUyku boşluğu uzunluğu (saat, katılımcı ortancası):")
print(p["sleep_gap_h"].describe().round(2))
print("\nGece kapsama oranı (uyku boşluğu bulunan gün payı):")
print(p["night_cover"].describe().round(2))
print("Kapsama < 0.5 olan katılımcı:", int((p["night_cover"] < 0.5).sum()))
print("\nGece oranı yüksek çıkan 8 katılımcı:")
print(p.loc[["P270", "P021", "P192", "P199", "P049", "P218", "P274", "P053"],
            ["n_nights", "n_days", "bedtime", "waketime", "sleep_gap_h"]].round(1))