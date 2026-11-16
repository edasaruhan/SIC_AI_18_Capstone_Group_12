"""7. adım: günlük ana tablo (katılımcı x gün). Risk skorunun ve modellerin tek girdisi.

Çıktı: master_daily.parquet
"""
import pandas as pd
from common import PROCESSED, RISK_GROUPS, load_ok_participants, load_sleep_windows, window_overlap_s

MESSAGING_CATEGORIES = ["mesajlasma", "iletisim", "eposta"]
MIN_APP_H_FOR_AS2 = 0.1     # daha az kullanımda dakikadaki uygulama geçişi (AS2) anlamsız büyür
LOOP_COLS = ["n_bouts", "n_reentry", "longest_bout_min"]


def load_events(windows):
    events = pd.read_parquet(PROCESSED / "app_events.parquet")
    events = events[(events["exclude"] == 0) & events["pid"].isin(load_ok_participants()["pid"])].copy()
    events.loc[events["category"].isin(MESSAGING_CATEGORIES), "risk_group"] = "iletisim"
    events = events.merge(windows[["pid", "bed_min", "wake_min"]], on="pid")
    end = events["ts"] + pd.to_timedelta(events["dwell_s"], unit="s")
    events["date"] = events["ts"].dt.normalize()
    events["dwell_h"] = events["dwell_s"] / 3600
    events["sleep_h"] = sum(window_overlap_s(events["ts"], end, events["bed_min"], events["wake_min"], shift)
                            for shift in (-1, 0)) / 3600
    return events


def hours_by_group(events, value, suffix):
    table = events.pivot_table(index=["pid", "date"], columns="risk_group", values=value,
                               aggfunc="sum", fill_value=0)
    return table.reindex(columns=RISK_GROUPS, fill_value=0).add_suffix(suffix)


def app_features_per_day(events):
    totals = events.groupby(["pid", "date"]).agg(n_events=("dwell_h", "size"), app_h=("dwell_h", "sum"),
                                                 sleep_use_h=("sleep_h", "sum"))
    switches_per_min = totals["n_events"] / (totals["app_h"] * 60)
    totals["as2"] = switches_per_min.where(totals["app_h"] >= MIN_APP_H_FOR_AS2)
    return (totals.join(hours_by_group(events, "dwell_h", "_h"))
            .join(hours_by_group(events, "sleep_h", "_sleep_h")).reset_index())


def main():
    windows = load_sleep_windows()
    apps = app_features_per_day(load_events(windows))
    screen = pd.read_parquet(PROCESSED / "daily_features.parquet").rename(columns={"short_share": "sess_short_share"})
    loops = pd.read_parquet(PROCESSED / "loop_daily.parquet").drop(columns="risky_h")

    master = screen.merge(apps, on=["pid", "date"], how="left").merge(loops, on=["pid", "date"], how="left")
    no_use_is_zero = [c for c in apps.columns if c not in ("pid", "date", "as2")] + LOOP_COLS
    master[no_use_is_zero] = master[no_use_is_zero].fillna(0)

    night = pd.read_csv(PROCESSED / "night_activity.csv")
    master["night_suspect"] = master["pid"].isin(night.loc[night["night_suspect"], "pid"])
    master = master.merge(windows[["pid", "reliable"]].rename(columns={"reliable": "sleep_window_reliable"}),
                          on="pid", how="left")
    master.to_parquet(PROCESSED / "master_daily.parquet")

    kept = master[~master["night_suspect"]]
    print("Katılımcı-gün:", len(master), "| şüpheli hariç:", len(kept), "| katılımcı:", kept["pid"].nunique())
    print("\nGünlük ortalama saat, risk grubuna göre:")
    print(kept[[f"{g}_h" for g in RISK_GROUPS]].mean().round(3))
    print("\nUyku penceresindeki ortalama saat/gün:")
    print(kept[[f"{g}_sleep_h" for g in RISK_GROUPS]].mean().round(3))


if __name__ == "__main__":
    main()
