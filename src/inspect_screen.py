from pathlib import Path
import pandas as pd

DATA = Path("mobile_phone_use/data")
files = sorted(DATA.glob("*.csv"))

cols = ["uuid", "ts", "ts_raw", "sensor_id", "DayOfWeek", "HourOfDay",
        "WorkingDay", "Screen_Value", "App_Value", "Charging_Value"]
df = pd.read_csv(files[0], usecols=cols, low_memory=False)

print("ts biçimi örnek:", df["ts"].iloc[0])
print("Satırlar zamana göre sıralı mı:", df["ts_raw"].is_monotonic_increasing)

scr = df[df.sensor_id == "Screen"]
print("\nScreen_Value:")
print(scr["Screen_Value"].value_counts(dropna=False))

app = df[df.sensor_id == "App"]
print("\nApp_Value (en sık 10):")
print(app["App_Value"].value_counts().head(10))

chg = df[df.sensor_id == "Charging"]
print("\nCharging_Value:")
print(chg["Charging_Value"].value_counts(dropna=False))