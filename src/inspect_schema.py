from pathlib import Path
import pandas as pd

DATA = Path("mobile_phone_use/data")
files = sorted(DATA.glob("*.csv"))
print("Katılımcı dosyası sayısı:", len(files))

df = pd.read_csv(files[0], low_memory=False)
print("Şekil:", df.shape)
print("Sütunlar:", list(df.columns))
print(df.dtypes)

if "sensor_id" in df.columns:
    print(df["sensor_id"].value_counts())