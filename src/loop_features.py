"""6. adım: riskli kategoride (kumar, oyun, sosyal/video) döngü özellikleri.

Nöbet      : araya en fazla MERGE_GAP_S saniye giren ardışık riskli uygulama olayları.
Geri dönüş : bir nöbet bittikten sonra en geç REENTRY_MAX_S içinde başlayan yeni nöbet.
Çıktı      : loop_daily.parquet (pid, date, n_bouts, n_reentry, longest_bout_min, risky_h)
"""
import pandas as pd
from common import PROCESSED, RISKY_GROUPS, load_ok_participants

MERGE_GAP_S = 120          # kısa bölünmeler (bildirime bakıp dönmek) aynı nöbet sayılır
REENTRY_MAX_S = 15 * 60


def build_bouts(events):
    risky = events[(events["exclude"] == 0) & events["risk_group"].isin(RISKY_GROUPS)]
    risky = risky.sort_values(["pid", "ts"]).copy()
    risky["end"] = risky["ts"] + pd.to_timedelta(risky["dwell_s"], unit="s")
    gap_s = (risky["ts"] - risky.groupby("pid")["end"].shift()).dt.total_seconds()
    risky["bout_id"] = (gap_s.isna() | (gap_s > MERGE_GAP_S)).cumsum()

    bouts = risky.groupby(["pid", "bout_id"]).agg(start=("ts", "min"), end=("end", "max"),
                                                  active_s=("dwell_s", "sum")).reset_index()
    since_previous_s = (bouts["start"] - bouts.groupby("pid")["end"].shift()).dt.total_seconds()
    bouts["reentry"] = since_previous_s.le(REENTRY_MAX_S)
    bouts["date"] = bouts["start"].dt.normalize()
    return bouts


def daily_loop_features(bouts):
    return bouts.groupby(["pid", "date"]).agg(
        n_bouts=("bout_id", "size"), n_reentry=("reentry", "sum"),
        longest_bout_min=("active_s", lambda s: s.max() / 60),
        risky_h=("active_s", lambda s: s.sum() / 3600)).reset_index()


def main():
    events = pd.read_parquet(PROCESSED / "app_events.parquet")
    events = events[events["pid"].isin(load_ok_participants()["pid"])]
    bouts = build_bouts(events)
    daily = daily_loop_features(bouts)
    daily.to_parquet(PROCESSED / "loop_daily.parquet")

    print("Nöbet:", len(bouts), "| nöbet içeren katılımcı-gün:", len(daily))
    print(daily.drop(columns=["pid", "date"]).describe(percentiles=[.5, .9, .99]).round(2))


if __name__ == "__main__":
    main()
