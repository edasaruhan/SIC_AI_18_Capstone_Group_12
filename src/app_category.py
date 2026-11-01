from pathlib import Path
import pandas as pd

OUT = Path("data/processed")
t = pd.read_csv(OUT / "app_counts.csv", index_col=0)
m = pd.read_csv("config/app_categories.csv").set_index("package")

PREFIX_RULES = [
    ("com.king.", "oyun", "oyun"),
    ("com.supercell.", "oyun", "oyun"),
    ("com.midasplayer", "oyun", "oyun"),
    ("com.zynga.", "oyun", "oyun"),
    ("com.ea.game", "oyun", "oyun"),
    ("com.etermax.", "oyun", "oyun"),
    ("com.nianticlabs.", "oyun", "oyun"),
]


def classify(pkg):
    if pkg in m.index:
        r = m.loc[pkg]
        return r["category"], r["risk_group"], int(r["exclude"]), "tam"
    if pkg.startswith("org.pielot"):
        return "calisma_uygulamasi", "dusuk", 1, "kural"
    for pre, cat, grp in PREFIX_RULES:
        if pkg.startswith(pre):
            return cat, grp, 0, "kural"
    return "bilinmeyen", "genel_web", 0, "yok"


res = pd.DataFrame([classify(p) for p in t.index], index=t.index,
                   columns=["category", "risk_group", "exclude", "how"])
t = t.join(res)
t.to_csv(OUT / "app_counts_labeled.csv")

tot = t["events"].sum()
print("Olay payı, eşleme yöntemine göre:")
print((t.groupby("how")["events"].sum() / tot).round(3))
print("\nOlay payı, risk grubuna göre (dışlananlar dahil):")
print((t.groupby("risk_group")["events"].sum() / tot).round(3))
print("\nDışlanan (exclude=1) olay payı:", round(t.loc[t["exclude"] == 1, "events"].sum() / tot, 3))
print("\nEşlenmemiş en sık 25 uygulama:")
print(t[t["how"] == "yok"].sort_values("events", ascending=False).head(25)[["events", "users"]].to_string())