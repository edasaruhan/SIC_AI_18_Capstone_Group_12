from pathlib import Path
import pandas as pd

DATA = Path("mobile_phone_use/data")
files = sorted(DATA.glob("*.csv"))
cols = ["sensor_id", "Esm_Responded", "Esm_Valence", "Esm_TiredAwake",
        "Esm_TenseCalm", "Esm_Bored", "Esm_Distract", "Esm_Dim", "Esm_Check"]
df = pd.read_csv(files[0], usecols=cols, low_memory=False)
e = df[df["sensor_id"] == "Esm"]
print("Esm satırı:", len(e))
pd.set_option("display.width", 200)
print(e.drop(columns="sensor_id").describe(include="all").T)
print("\nEsm_Responded:")
print(e["Esm_Responded"].value_counts(dropna=False))

pi = pd.read_csv("mobile_phone_use/pinfo.csv")
print("\npinfo şekli:", pi.shape)
print("Sütun ve dolu satır sayısı:")
print(pi.notna().sum().to_string())