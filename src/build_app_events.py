"""5c. adım: her uygulama olayına süre (dwell) ve kategori ekler.

Süre = bir sonraki uygulama olayına veya ekranın kapanmasına kadar geçen zaman.
Çıktı: app_events.parquet (pid, ts, dwell_s, category, risk_group, exclude)
"""
import pandas as pd
from common import NON_APP_PREFIXES, PROCESSED, STUDY_PREFIX, load_ok_participants, raw_files

MAX_DWELL_S = 7200
NOT_AN_APP = ("sistem_ozel", "dusuk", 1)
UNKNOWN_APP = ("bilinmeyen", "genel_web", 0)
LABEL_COLS = ["category", "risk_group", "exclude"]


def app_events_with_dwell(path):
    df = pd.read_csv(path, usecols=["ts", "ts_raw", "sensor_id", "Screen_Value", "App_Value"], low_memory=False)
    is_app = df["sensor_id"] == "App"
    is_screen_off = (df["sensor_id"] == "Screen") & (df["Screen_Value"] == "Off")
    timeline = df.loc[is_app | is_screen_off, ["ts", "ts_raw", "sensor_id", "App_Value"]].sort_values("ts_raw")
    until_next_s = (timeline["ts_raw"].shift(-1) - timeline["ts_raw"]) / 1000
    timeline["dwell_s"] = until_next_s.clip(upper=MAX_DWELL_S)
    return timeline[(timeline["sensor_id"] == "App") & timeline["App_Value"].notna() & timeline["dwell_s"].notna()]


def main():
    ok_pids = set(load_ok_participants()["pid"])
    labeled = pd.read_csv(PROCESSED / "app_counts_labeled.csv", index_col=0)
    known = {pkg: (row.category, row.risk_group, int(row.exclude)) for pkg, row in labeled.iterrows()}

    def label(app):
        if app.startswith(NON_APP_PREFIXES) or app.startswith(STUDY_PREFIX):
            return NOT_AN_APP
        return known.get(app, UNKNOWN_APP)

    per_participant = []
    for pid, path in raw_files():
        if pid not in ok_pids:
            continue
        apps = app_events_with_dwell(path)
        labels = pd.DataFrame(apps["App_Value"].map(label).tolist(), index=apps.index, columns=LABEL_COLS)
        per_participant.append(apps[["ts", "dwell_s"]].join(labels).assign(pid=pid))

    events = pd.concat(per_participant, ignore_index=True)[["pid", "ts", "dwell_s"] + LABEL_COLS]
    events["ts"] = pd.to_datetime(events["ts"])
    events.to_parquet(PROCESSED / "app_events.parquet")

    kept = events[events["exclude"] == 0]
    print("Olay:", len(events), "| dışlanmayan:", len(kept))
    print("\nSüre payı, risk grubuna göre:")
    print((kept.groupby("risk_group")["dwell_s"].sum() / kept["dwell_s"].sum()).round(3))
    print("Üst sınıra takılan olay payı:", round((kept["dwell_s"] >= MAX_DWELL_S).mean(), 4))


if __name__ == "__main__":
    main()
