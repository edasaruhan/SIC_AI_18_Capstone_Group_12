"""10b. adım: dış geçerlilik (gün düzeyi, kişi-içi). Bir gecenin uyku penceresindeki kullanım
ve önceki günün risk skoru, ertesi günün öz-bildirim ruh halini öngörüyor mu?

Yöntem: değişkenler kişinin kendi ortalamasından arındırılır (kişi-içi etki); eğimin %95 güven
aralığı katılımcı düzeyinde yeniden örneklemeyle (bootstrap) bulunur.
Çıktı: reports/validity_daily.csv
"""
import numpy as np
import pandas as pd
from common import PROCESSED, REPORTS, load_sleep_windows, window_overlap_s

MIN_DAYS_PER_PERSON = 5
MIN_PARTICIPANTS = 20
N_BOOTSTRAP = 2000
MORNING_END_H = 12
MOODS = ["valence", "tired_awake", "tense_calm", "bored"]
RISK_COLS = ["risk", "sleep_pts", "cat_pts", "loop_pts", "screen_h"]
PREDICTORS = ["night_use_h"] + [f"{c}_dun" for c in RISK_COLS]


def night_use():
    """Gece başına uyku penceresindeki kullanım (saat). Gece, sabahına uyanılan tarihle etiketlenir."""
    events = pd.read_parquet(PROCESSED / "app_events.parquet")
    windows = load_sleep_windows()
    events = events[events["exclude"] == 0].merge(windows[windows["reliable"]], on="pid")
    end = events["ts"] + pd.to_timedelta(events["dwell_s"], unit="s")
    event_day = events["ts"].dt.normalize()
    per_shift = []
    for day_shift in (-1, 0):
        overlap_s = window_overlap_s(events["ts"], end, events["bed_min"], events["wake_min"], day_shift)
        per_shift.append(pd.DataFrame({"pid": events["pid"].values,
                                       "date": (event_day + pd.Timedelta(days=day_shift + 1)).values,
                                       "night_use_h": overlap_s / 3600}))
    return pd.concat(per_shift).groupby(["pid", "date"], as_index=False)["night_use_h"].sum()


def daily_mood():
    esm = pd.read_parquet(PROCESSED / "esm.parquet")
    esm["date"] = esm["ts"].dt.normalize()
    mood = esm.groupby(["pid", "date"])[MOODS].mean().reset_index()
    morning = (esm[esm["ts"].dt.hour < MORNING_END_H].groupby(["pid", "date"])["tired_awake"].mean()
               .rename("tired_awake_sabah").reset_index())
    return mood.merge(morning, on=["pid", "date"], how="left")


def previous_day_risk():
    risk = pd.read_parquet(PROCESSED / "risk_daily.parquet")[["pid", "date"] + RISK_COLS]
    risk["date"] = risk["date"] + pd.Timedelta(days=1)
    return risk.rename(columns={c: f"{c}_dun" for c in RISK_COLS})


def within_person(data, predictor, outcome):
    """Yeterli günü olan kişilerde, iki değişkenin kişi ortalamasından sapmaları."""
    sample = data[["pid", predictor, outcome]].dropna()
    sample = sample[sample.groupby("pid")[predictor].transform("size") >= MIN_DAYS_PER_PERSON]
    by_person = sample.groupby("pid")
    return sample.assign(x=sample[predictor] - by_person[predictor].transform("mean"),
                         y=sample[outcome] - by_person[outcome].transform("mean"))


def slope_with_ci(sample, rng):
    """Kişi-içi eğim ve katılımcıları yeniden örnekleyerek bulunan %95 güven aralığı."""
    sums = sample.assign(xy=sample["x"] * sample["y"], xx=sample["x"] ** 2).groupby("pid")[["xy", "xx"]].sum().values
    slope = (sample["x"] * sample["y"]).sum() / (sample["x"] ** 2).sum()
    draws = rng.integers(0, len(sums), (N_BOOTSTRAP, len(sums)))
    boot = sums[draws, 0].sum(axis=1) / sums[draws, 1].sum(axis=1)
    return slope, *np.nanpercentile(boot, [2.5, 97.5])


def main():
    risk = previous_day_risk()
    data = daily_mood().merge(night_use(), on=["pid", "date"], how="left").merge(risk, on=["pid", "date"], how="left")
    data = data[data["pid"].isin(risk["pid"].unique())]       # yalnızca analizdeki katılımcılar
    rng = np.random.default_rng(0)
    rows = []
    for outcome in MOODS + ["tired_awake_sabah"]:
        for predictor in PREDICTORS:
            sample = within_person(data, predictor, outcome)
            if sample["pid"].nunique() < MIN_PARTICIPANTS:
                continue
            slope, low, high = slope_with_ci(sample, rng)
            rows.append(dict(sonuc=outcome, yordayici=predictor, n_gun=len(sample), n_kisi=sample["pid"].nunique(),
                             egim=slope, ci_alt=low, ci_ust=high, r_kisi_ici=sample["x"].corr(sample["y"]),
                             anlamli=bool(low > 0 or high < 0)))
    results = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    results.round(4).to_csv(REPORTS / "validity_daily.csv", index=False)

    print("Ruh hali bildirimi olan katılımcı-gün:", len(data), "| katılımcı:", data["pid"].nunique())
    print(results.round(4).to_string(index=False))
    print("\nOkuma: egim = yordayıcı kişinin kendi ortalamasının 1 birim üstündeyken sonuçtaki değişim.")


if __name__ == "__main__":
    main()
