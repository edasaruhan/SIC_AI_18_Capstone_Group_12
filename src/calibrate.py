import copy
from pathlib import Path
import numpy as np
import pandas as pd
from risk import DEFAULT, compute_components

OUT = Path("data/processed")
master = pd.read_parquet(OUT / "master_daily.parquet")
v = master[~master["night_suspect"]].copy()

r0 = compute_components(v)
base_r = r0["risk"]

# 1) Yüzdelik tabanlı eşikler
th = base_r.quantile([0.90, 0.97, 0.995])
print("Önerilen eşikler (risk, 0-1): Dikkat >=", round(th.iloc[0], 3),
      "| Yüksek >=", round(th.iloc[1], 3), "| Kritik >=", round(th.iloc[2], 3))
r0["level"] = pd.cut(base_r, [-1, th.iloc[0], th.iloc[1], th.iloc[2], 2],
                     labels=["Normal", "Dikkat", "Yüksek", "Kritik"], right=False)

# 2) Uyarı yükü (kullanıcı başına Dikkat+ gün payı)
flag = (r0["level"] != "Normal")
per = flag.groupby(r0["pid"]).mean()
print("\nKatılımcı başına Dikkat veya üstü gün payı:")
print(per.describe(percentiles=[0.5, 0.9]).round(3))
print("En az bir Kritik günü olan katılımcı payı:",
      round(r0.loc[r0["level"] == "Kritik", "pid"].nunique() / r0["pid"].nunique(), 3))

print("\nSeviyeye göre bileşen ortalamaları (puan):")
print(r0.groupby("level", observed=True)[["cat_pts", "sleep_pts", "dev_pts", "loop_pts"]].mean().round(1))

# 3) Duyarlılık: parametreleri ±%25 rastgele oynat
rng = np.random.default_rng(0)


def perturb(p):
    q = copy.deepcopy(p)
    for k in q["PTS"]:
        q["PTS"][k] *= rng.uniform(0.75, 1.25)
    for k in ("H_CAT", "H_SLEEP"):
        q[k] *= rng.uniform(0.75, 1.25)
    for grp in ("W_CAT", "W_SLEEP"):
        for k in q[grp]:
            q[grp][k] *= rng.uniform(0.75, 1.25)
    return q


def top10(s):
    return set(s.nlargest(int(0.10 * len(s))).index)


A = top10(base_r)
res = []
for _ in range(30):
    r = compute_components(v, perturb(DEFAULT))["risk"]
    B = top10(r)
    res.append((base_r.corr(r, method="spearman"), len(A & B) / len(A | B)))
res = pd.DataFrame(res, columns=["spearman", "top10_ortusme"])
print("\nDuyarlılık (30 rastgele parametre seti):")
print(res.describe().loc[["mean", "min"]].round(3))