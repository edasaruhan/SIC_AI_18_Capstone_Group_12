from pathlib import Path
import pandas as pd

DATA = Path("mobile_phone_use/data")
OUT = Path("data/processed")
MAX_DWELL_S = 7200

ok = set(pd.read_csv(OUT / "participants_ok.csv")["pid"])
lab = pd.read_csv(OUT / "app_counts_labeled.csv", index_col=0)
cat = lab[["category", "risk_group", "exclude"]].to_dict("index")


def label(app):
    if app and app[0] in "+=-" or app.startswith("org.pielot"):
        return ("sistem_ozel", "dusuk", 1)
    r = cat.get(app)
    if r:
        return (r["category"], r["risk_group"], int(r["exclude"]))
    return ("bilinmeyen", "genel_web", 0)


rows = []
for i, f in enumerate(sorted(DATA.glob("*.csv"))):
    pid = f"P{i:03d}"
    if pid not in ok:
        continue
    df = pd.read_csv(f, usecols=["ts", "ts_raw", "sensor_id", "Screen_Value", "App_Value"],
                     low_memory=False)
    m = (df["sensor_id"] == "App") | ((df["sensor_id"] == "Screen") & (df["Screen_Value"] == "Off"))
    d = df.loc[m, ["ts", "ts_raw", "sensor_id", "App_Value"]].sort_values("ts_raw").reset_index(drop=True)
    d["next_raw"] = d["ts_raw"].shift(-1)
    d["dwell_s"] = ((d["next_raw"] - d["ts_raw"]) / 1000).clip(upper=MAX_DWELL_S)
    a = d[(d["sensor_id"] == "App") & d["App_Value"].notna() & d["dwell_s"].notna()].copy()
    labs = pd.DataFrame(a["App_Value"].map(label).tolist(), index=a.index,
                        columns=["category", "risk_group", "exclude"])
    a = a.join(labs)
    a["pid"] = pid
    rows.append(a[["pid", "ts", "dwell_s", "category", "risk_group", "exclude"]])

ev = pd.concat(rows, ignore_index=True)
ev["ts"] = pd.to_datetime(ev["ts"])
ev.to_parquet(OUT / "app_events.parquet")

v = ev[ev["exclude"] == 0]
print("Olay:", len(ev), "| dışlanmayan:", len(v))
print("\nDwell süresi payı, risk grubuna göre:")
print((v.groupby("risk_group")["dwell_s"].sum() / v["dwell_s"].sum()).round(3))
print("\nOrtanca dwell (sn):", round(v["dwell_s"].median(), 1))
print("Üst sınıra takılan olay payı:", round((v["dwell_s"] >= MAX_DWELL_S).mean(), 4))
g = v[v["risk_group"].isin(["oyun", "kumar"])]
print("Oyun/kumar uygulaması kullanan katılımcı:", g["pid"].nunique(), "/", v["pid"].nunique())