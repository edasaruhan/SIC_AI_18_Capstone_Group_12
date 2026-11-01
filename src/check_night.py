from pathlib import Path
import pandas as pd

OUT = Path("data/processed")
d = pd.read_parquet(OUT / "daily_features.parquet")
d = d[d["screen_h"] >= 0.5]

p = d.groupby("pid").agg(median_night=("night_ratio", "median"),
                         days=("screen_h", "size"),
                         screen_h=("screen_h", "median"))

print("Katılımcı bazında medyan gece oranı dağılımı:")
print(p["median_night"].describe(percentiles=[0.5, 0.75, 0.9, 0.95]).round(2))
print("\nMedyan gece oranı > 0.30 olan katılımcı sayısı:", int((p["median_night"] > 0.30).sum()))
print("Gece oranı > 0.50 olan gün payı:", round((d["night_ratio"] > 0.50).mean(), 3))
print("\nEn yüksek 8 katılımcı (pid, medyan gece oranı, medyan günlük ekran saati):")
print(p.sort_values("median_night", ascending=False).head(8).round(2))