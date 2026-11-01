from pathlib import Path
import pandas as pd
from sessions import sessionize

COLS = ["uuid", "ts", "ts_raw", "sensor_id", "Screen_Value", "App_Value",
        "Charging_Value", "Acc_Avg", "Acc_Max", "UserAct_Type", "UserAct_p"]

files = sorted(Path("mobile_phone_use/data").glob("*.csv"))
df = pd.read_csv(files[0], usecols=COLS, low_memory=False)
df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts_raw").reset_index(drop=True)
s = sessionize(df)

print("UserAct_Type dağılımı:")
print(df.loc[df.sensor_id == "UserAct", "UserAct_Type"].astype(str).value_counts().head(8))

acc = df.loc[df.sensor_id == "Acc", ["ts", "Acc_Max"]]
act = df.loc[df.sensor_id == "UserAct", ["ts", "UserAct_Type"]]


def summarize(row):
    a = acc[(acc.ts >= row.start) & (acc.ts <= row.end)]
    u = act[(act.ts >= row.start) & (act.ts <= row.end)]
    still = u["UserAct_Type"].astype(str).str.contains("still").mean() if len(u) else float("nan")
    return pd.Series({"acc_std": a["Acc_Max"].std(), "still_payi": still, "acc_olay": len(a)})


groups = {
    "kısa (<5 dk, kilidi açık)": s[(s.duration_s < 300) & s.unlocked],
    "uzun (>2 saat)": s[s.duration_s > 7200],
}
for name, g in groups.items():
    r = g.apply(summarize, axis=1)
    print(f"\n{name} - oturum sayısı: {len(g)}")
    print(r.median().round(3))