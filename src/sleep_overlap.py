"""4b. adım: her gece, ekranın uyku penceresi içinde açık kaldığı süre.

Çıktı: sleep_overlap.parquet (pid, night, reliable, overlap_min)
"""
import pandas as pd
from common import PROCESSED, load_sleep_windows, load_valid_sessions, window_overlap_s


def main():
    windows = load_sleep_windows()
    sessions = load_valid_sessions().merge(windows, on="pid")
    session_day = sessions["start"].dt.normalize()
    per_night = []
    for day_shift in (-1, 0):
        per_night.append(pd.DataFrame({
            "pid": sessions["pid"], "night": session_day + pd.Timedelta(days=day_shift),
            "reliable": sessions["reliable"],
            "overlap_s": window_overlap_s(sessions["start"], sessions["end_local"],
                                          sessions["bed_min"], sessions["wake_min"], day_shift)}))
    nights = pd.concat(per_night).groupby(["pid", "night", "reliable"], as_index=False)["overlap_s"].sum()
    nights["overlap_min"] = nights["overlap_s"] / 60
    nights.drop(columns="overlap_s").to_parquet(PROCESSED / "sleep_overlap.parquet")

    print("Güvenilir pencereli katılımcı:", int(windows["reliable"].sum()), "/", len(windows))
    for reliable, group in nights.groupby("reliable"):
        print(f"Güvenilir pencere = {reliable}: {len(group)} gece | "
              f">=30 dk çakışma payı: {(group['overlap_min'] >= 30).mean():.2f}")


if __name__ == "__main__":
    main()
