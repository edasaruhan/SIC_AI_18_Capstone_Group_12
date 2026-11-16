"""4c. adım: "gece ekranı hep açık" görünen katılımcıları bulur.

Uyku penceresi güvenilmez çıkan katılımcılarda ekran gece açık ama uygulama olayı yoksa
(gece/gündüz olay oranı düşükse) bu kullanım değil cihaz davranışıdır; katılımcı şüpheli
işaretlenir. Karşılaştırma için güvenilir pencereli rastgele bir grup da ölçülür.
Çıktı: night_activity.csv
"""
import numpy as np
import pandas as pd
from common import (PROCESSED, WINDOW_ANCHOR_H, is_real_app, load_ok_participants,
                    load_sleep_windows, raw_files)

SUSPECT_RATIO = 0.30        # gece olay hızı gündüzün bu oranının altındaysa şüpheli
MIN_NIGHT_H = 0.5           # oran için gereken en az uyku penceresi ekran süresi
COMPARISON_SAMPLE = 80
MINUTES_PER_DAY = 1440
WINDOW_ANCHOR_MIN = WINDOW_ANCHOR_H * 60


def pick_participants(windows):
    """Güvenilmez pencerelilerin hepsi + güvenilirlerden rastgele bir karşılaştırma grubu."""
    rng = np.random.default_rng(0)
    reliable = windows.loc[windows["reliable"], "pid"]
    sample = rng.choice(reliable, size=min(COMPARISON_SAMPLE, len(reliable)), replace=False)
    return set(windows.loc[~windows["reliable"], "pid"]) | set(sample)


def app_event_minutes(pids):
    """Seçilen katılımcıların uygulama olayları: gün içi dakika (0-1439)."""
    events = []
    for pid, path in raw_files():
        if pid not in pids:
            continue
        df = pd.read_csv(path, usecols=["ts", "sensor_id", "App_Value"], low_memory=False)
        apps = df[df["sensor_id"] == "App"].dropna(subset=["App_Value"])
        ts = pd.to_datetime(apps.loc[is_real_app(apps["App_Value"]), "ts"])
        events.append(pd.DataFrame({"pid": pid, "minute": (ts.dt.hour * 60 + ts.dt.minute).values}))
    return pd.concat(events)


def count_night_events(events, windows):
    windows = windows.assign(
        bed_clock=(windows["bed_min"] + WINDOW_ANCHOR_MIN) % MINUTES_PER_DAY,
        wake_clock=(windows["wake_min"] + WINDOW_ANCHOR_MIN) % MINUTES_PER_DAY)
    ev = events.merge(windows[["pid", "bed_clock", "wake_clock"]], on="pid")
    after_bed, before_wake = ev["minute"] >= ev["bed_clock"], ev["minute"] < ev["wake_clock"]
    wraps_midnight = ev["bed_clock"] > ev["wake_clock"]
    ev["night"] = np.where(wraps_midnight, after_bed | before_wake, after_bed & before_wake)
    return ev.groupby("pid").agg(n_app=("night", "size"), n_night=("night", "sum"))


def main():
    windows = load_sleep_windows()
    pids = pick_participants(windows) & set(load_ok_participants()["pid"])
    result = count_night_events(app_event_minutes(pids), windows)

    sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    sessions = sessions[sessions["valid"] & sessions["pid"].isin(result.index)]
    overlap = pd.read_parquet(PROCESSED / "sleep_overlap.parquet")
    result["total_h"] = sessions.groupby("pid")["duration_s"].sum() / 3600
    result["night_h"] = overlap.groupby("pid")["overlap_min"].sum() / 60
    result = result.join(windows.set_index("pid")["reliable"])

    night_rate = result["n_night"] / result["night_h"].where(result["night_h"] > MIN_NIGHT_H)
    day_rate = (result["n_app"] - result["n_night"]) / (result["total_h"] - result["night_h"])
    result["ratio"] = night_rate / day_rate
    result["night_suspect"] = ~result["reliable"].astype(bool) & (result["ratio"] < SUSPECT_RATIO)
    result[["reliable", "night_h", "ratio", "night_suspect"]].to_csv(PROCESSED / "night_activity.csv")

    for reliable, group in result.groupby("reliable"):
        print(f"Güvenilir pencere = {reliable}: {len(group)} katılımcı | "
              f"gece/gündüz olay oranı (ortanca): {group['ratio'].median():.2f} | "
              f"oranı < {SUSPECT_RATIO} olanların payı: {(group['ratio'] < SUSPECT_RATIO).mean():.2f}")
    print("Şüpheli katılımcı:", int(result["night_suspect"].sum()))


if __name__ == "__main__":
    main()
