from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/processed")
m = pd.read_parquet(OUT / "master_daily.parquet").sort_values(["pid", "date"]).reset_index(drop=True)

# ---- parametreler: ekiple birlikte ayarlanacak ----
H_CAT = 3.0
W_CAT = {"kumar_h": 1.0, "oyun_h": 1.0, "sosyal_video_h": 0.7, "genel_web_h": 0.3}
H_SLEEP = 1.0
W_SLEEP = {"kumar_sleep_h": 1.2, "oyun_sleep_h": 1.2, "sosyal_video_sleep_h": 1.0,
           "genel_web_sleep_h": 0.6, "dusuk_sleep_h": 0.6}
BASE_DAYS = 14

m["cat_pts"] = 20 * np.clip(sum(w * m[c] for c, w in W_CAT.items()) / H_CAT, 0, 1)
m["sleep_pts"] = 25 * np.clip(sum(w * m[c] for c, w in W_SLEEP.items()) / H_SLEEP, 0, 1)

# kişisel taban (ilk BASE_DAYS gün)
m["day_idx"] = m.groupby("pid").cumcount()
base = (m[m["day_idx"] < BASE_DAYS].groupby("pid")["screen_h"]
        .agg(base_med="median", base_mad=lambda x: (x - x.median()).abs().median()))
m = m.merge(base, on="pid", how="left")
z = (m["screen_h"] - m["base_med"]) / (1.4826 * m["base_mad"] + 0.5)
m["dev_pts"] = np.where(m["day_idx"] >= BASE_DAYS, 20 * np.clip(z / 3, 0, 1), 0.0)

# döngü
as2 = m["as2"].where(m["app_h"] >= 0.1)
loop_a = np.clip((as2 - 1.0) / 2.0, 0, 1).fillna(0)
loop_b = np.clip((m["sess_short_share"] - 0.4) / 0.4, 0, 1).fillna(0)
m["loop_pts"] = 15 * np.clip(0.5 * loop_a + 0.5 * loop_b, 0, 1)

m["raw"] = m[["cat_pts", "sleep_pts", "dev_pts", "loop_pts"]].sum(axis=1)
m["risk"] = m["raw"] / 80
m["level"] = pd.cut(m["risk"], [-0.01, 0.40, 0.70, 0.90, 1.01],
                    labels=["Normal", "Dikkat", "Yüksek", "Kritik"], right=False)
m.to_parquet(OUT / "risk_daily.parquet")

pd.set_option("display.width", 200)
v = m[~m["night_suspect"]]
print("Gün:", len(v), "| katılımcı:", v["pid"].nunique())
print("\nSeviye payları:")
print(v["level"].value_counts(normalize=True).reindex(["Normal", "Dikkat", "Yüksek", "Kritik"]).round(3))
print("\nBileşen ortalamaları (puan):")
print(v[["cat_pts", "sleep_pts", "dev_pts", "loop_pts"]].mean().round(2))
hi = v[v["level"].isin(["Yüksek", "Kritik"])]
print("\nYüksek/Kritik günü olan katılımcı payı:", round(hi["pid"].nunique() / v["pid"].nunique(), 3))
print("Risk ile ekran saati korelasyonu:", round(v[["risk", "screen_h"]].corr().iloc[0, 1], 2))
print("\nEn yüksek 5 gün:")
print(v.nlargest(5, "risk")[["cat_pts", "sleep_pts", "dev_pts", "loop_pts", "risk",
                             "oyun_sleep_h", "screen_h"]].round(2).to_string(index=False))