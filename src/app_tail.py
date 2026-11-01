from pathlib import Path
import pandas as pd

OUT = Path("data/processed")
t = pd.read_csv(OUT / "app_counts.csv", index_col=0)
m = pd.read_csv("config/app_categories.csv")

rest = t[~t.index.isin(m["package"]) & ~t.index.str.startswith("org.pielot")]
print("Eşlenmemiş uygulamalar, olay sayısına göre ilk 60:")
print(rest.sort_values("events", ascending=False).head(60).to_string())

pat = "game|casino|poker|slot|bingo|saga|candy|puzzle|racing|clash|farm|jackpot|lotto|quiniela|sportsbook|bet365|betway"
g = rest[rest.index.str.contains(pat, case=False)]
print("\nAnahtar kelimeyle oyun/kumar adayı:", len(g), "uygulama")
print(g.sort_values("events", ascending=False).head(40).to_string())