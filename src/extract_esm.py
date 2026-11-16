"""10a. adım: ham dosyalardan yanıtlanmış ESM (anlık ruh hali anketi) satırlarını çıkarır.

Çıktı: esm.parquet (pid, ts, valence, tired_awake, tense_calm, bored, distract)
"""
import pandas as pd
from common import PROCESSED, raw_files

ESM_COLUMNS = {"Esm_Valence": "valence", "Esm_TiredAwake": "tired_awake", "Esm_TenseCalm": "tense_calm",
               "Esm_Bored": "bored", "Esm_Distract": "distract"}
WANTED = {"ts", "sensor_id", *ESM_COLUMNS}


def answered_esm(path):
    df = pd.read_csv(path, usecols=lambda col: col in WANTED, low_memory=False)
    answers = [col for col in ESM_COLUMNS if col in df.columns]
    esm = df[df["sensor_id"].astype(str).str.startswith("Esm")].copy()
    esm[answers] = esm[answers].apply(pd.to_numeric, errors="coerce")
    esm = esm.dropna(subset=answers, how="all")
    esm["ts"] = pd.to_datetime(esm["ts"], errors="coerce")
    return esm.dropna(subset=["ts"]).rename(columns=ESM_COLUMNS).drop(columns="sensor_id")


def main():
    per_participant = [answered_esm(path).assign(pid=pid) for pid, path in raw_files()]
    esm = pd.concat(per_participant, ignore_index=True).drop_duplicates(["pid", "ts"])
    esm = esm[["pid", "ts"] + [c for c in ESM_COLUMNS.values() if c in esm.columns]]
    esm.to_parquet(PROCESSED / "esm.parquet")

    print("Yanıtlanmış ESM:", len(esm), "| katılımcı:", esm["pid"].nunique())
    print(esm.drop(columns=["pid", "ts"]).describe().round(2))


if __name__ == "__main__":
    main()
