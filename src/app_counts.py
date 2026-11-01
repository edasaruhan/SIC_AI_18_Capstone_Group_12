from collections import Counter
from pathlib import Path
import pandas as pd

DATA = Path("mobile_phone_use/data")
OUT = Path("data/processed")
ok = set(pd.read_csv(OUT / "participants_ok.csv")["pid"])

events, users = Counter(), Counter()
for i, f in enumerate(sorted(DATA.glob("*.csv"))):
    if f"P{i:03d}" not in ok:
        continue
    df = pd.read_csv(f, usecols=["sensor_id", "App_Value"], low_memory=False)
    a = df.loc[df["sensor_id"] == "App", "App_Value"].dropna()
    a = a[~a.str.startswith(("+", "=", "-")) & (a != "org.pielot.sns2")]
    vc = a.value_counts()
    events.update(vc.to_dict())
    users.update(vc.index)

table = pd.DataFrame({"events": pd.Series(events), "users": pd.Series(users)})
table = table.sort_values("users", ascending=False)
table.to_csv(OUT / "app_counts.csv")

print("Farklı uygulama:", len(table))
print("Toplam olay:", int(table["events"].sum()))
for k in (50, 100, 200):
    share = table.head(k)["events"].sum() / table["events"].sum()
    print(f"Kullanıcı sayısına göre en sık {k} uygulama, olayların payı: {share:.2f}")
print()
print(table.head(60).to_string())