"""5a. adım: hangi uygulama kaç olayda ve kaç katılımcıda görülüyor? (kategori eşlemesinin girdisi)

Çıktı: app_counts.csv
"""
from collections import Counter
import pandas as pd
from common import PROCESSED, is_real_app, load_ok_participants, raw_files


def main():
    ok_pids = set(load_ok_participants()["pid"])
    events, users = Counter(), Counter()
    for pid, path in raw_files():
        if pid not in ok_pids:
            continue
        df = pd.read_csv(path, usecols=["sensor_id", "App_Value"], low_memory=False)
        apps = df.loc[df["sensor_id"] == "App", "App_Value"].dropna()
        counts = apps[is_real_app(apps)].value_counts()
        events.update(counts.to_dict())
        users.update(counts.index)

    table = pd.DataFrame({"events": pd.Series(events), "users": pd.Series(users)})
    table = table.sort_values("users", ascending=False)
    table.to_csv(PROCESSED / "app_counts.csv")
    print("Farklı uygulama:", len(table), "| toplam olay:", int(table["events"].sum()))


if __name__ == "__main__":
    main()
