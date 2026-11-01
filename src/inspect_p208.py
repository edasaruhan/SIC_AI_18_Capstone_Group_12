from pathlib import Path
import pandas as pd
from sessions import load_participant, sessionize

files = sorted(Path("mobile_phone_use/data").glob("*.csv"))
df = load_participant(files[208])
s = sessionize(df)
s = s[s["valid"]].copy()

daily = s.groupby("date")["duration_s"].sum() / 3600
print("Günlük ekran saati (min / medyan / maks):",
      round(daily.min(), 1), round(daily.median(), 1), round(daily.max(), 1))
print("20 saati aşan gün sayısı:", int((daily > 20).sum()), "/", len(daily))

by_hour = (s.groupby("hour")["duration_s"].sum() / 3600).round(1)
print("\nBaşlangıç saatine göre toplam ekran saati:")
print(by_hour.to_string())

chg = (df[df.sensor_id == "Charging"][["ts", "Charging_Value"]]
       .sort_values("ts").rename(columns={"ts": "chg_ts"}))
s = pd.merge_asof(s.sort_values("start"), chg, left_on="start",
                  right_on="chg_ts", direction="backward")
print("\nŞarjda başlayan oturumların ekran süresi payı:",
      round(s.loc[s.Charging_Value == 1, "duration_s"].sum() / s["duration_s"].sum(), 2))

app = df[df.sensor_id == "App"]["App_Value"].dropna()
app = app[~app.str.startswith(("+", "=", "-"))]
app = app[app != "org.pielot.sns2"]
print("\nEn sık 8 uygulama (olay sayısı):")
print(app.value_counts().head(8))