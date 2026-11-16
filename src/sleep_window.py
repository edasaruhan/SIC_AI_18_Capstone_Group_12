"""4a. adım: her katılımcının kişisel uyku penceresini tahmin eder.

Yöntem: her gece, akşam-sabah arasında başlayan en uzun ekran-kapalı boşluk uyku sayılır;
katılımcının yatış ve kalkış saati bu boşlukların ortancasıdır.
Çıktı: sleep_windows.csv (bed_min / wake_min: gecenin 18:00'inden itibaren dakika)
"""
import pandas as pd
from common import PROCESSED, WINDOW_ANCHOR_H, load_valid_sessions

MIN_GAP_H, MAX_GAP_H = 3, 14
EVENING_START_H, MORNING_END_H = 18, 9    # uyku boşluğu bu saatler arasında başlamalı


def nightly_sleep_gaps(sessions):
    sessions = sessions.sort_values(["pid", "start"])
    gaps = pd.DataFrame({"pid": sessions["pid"], "gap_start": sessions["end_local"],
                         "gap_end": sessions.groupby("pid")["start"].shift(-1)}).dropna(subset=["gap_end"])
    gaps["gap_h"] = (gaps["gap_end"] - gaps["gap_start"]).dt.total_seconds() / 3600
    start_hour = gaps["gap_start"].dt.hour
    gaps = gaps[gaps["gap_h"].between(MIN_GAP_H, MAX_GAP_H)
                & ((start_hour >= EVENING_START_H) | (start_hour < MORNING_END_H))].copy()
    gaps["night"] = (gaps["gap_start"] - pd.Timedelta(hours=12)).dt.normalize()
    longest = gaps.sort_values("gap_h").groupby(["pid", "night"]).tail(1).copy()
    anchor = longest["night"] + pd.Timedelta(hours=WINDOW_ANCHOR_H)
    longest["bed_min"] = (longest["gap_start"] - anchor).dt.total_seconds() / 60
    longest["wake_min"] = (longest["gap_end"] - anchor).dt.total_seconds() / 60
    return longest


def as_clock(minutes_from_anchor):
    if pd.isna(minutes_from_anchor):
        return None
    minutes = int(round(minutes_from_anchor)) + WINDOW_ANCHOR_H * 60
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def main():
    sessions = load_valid_sessions()
    windows = nightly_sleep_gaps(sessions).groupby("pid").agg(
        n_nights=("gap_h", "size"), bed_min=("bed_min", "median"),
        wake_min=("wake_min", "median"), sleep_gap_h=("gap_h", "median"))
    n_days = sessions.groupby("pid")["start"].apply(lambda s: s.dt.normalize().nunique()).rename("n_days")
    windows = windows.join(n_days, how="right")
    windows["n_nights"] = windows["n_nights"].fillna(0)
    windows["night_cover"] = windows["n_nights"] / windows["n_days"]
    windows["bedtime"] = windows["bed_min"].map(as_clock)
    windows["waketime"] = windows["wake_min"].map(as_clock)
    windows.to_csv(PROCESSED / "sleep_windows.csv")

    print("Katılımcı:", len(windows), "| uyku boşluğu bulunan:", int((windows["n_nights"] > 0).sum()))
    print("Ortanca yatış:", as_clock(windows["bed_min"].median()),
          "| ortanca kalkış:", as_clock(windows["wake_min"].median()))
    print("\nUyku boşluğu (saat) ve gece kapsama oranı:")
    print(windows[["sleep_gap_h", "night_cover"]].describe().round(2))


if __name__ == "__main__":
    main()
