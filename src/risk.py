"""Günlük risk skoru (0-1): dört bileşenin puan toplamı / toplam puan.

kategori  : riskli kategorilerde geçen ağırlıklı süre
uyku      : kişinin kendi uyku penceresine denk gelen ağırlıklı kullanım
sapma     : kişinin ilk BASE_DAYS günündeki ekran süresine göre bugünkü fazlalık
döngü     : riskli kategoriye sık geri dönüş ve kesintisiz uzun nöbet
"""
import numpy as np
import pandas as pd

DEFAULT = {
    "PTS": {"cat": 20, "sleep": 25, "dev": 20, "loop": 15},
    # ağırlıklı saat toplamı H_CAT'e ulaşınca kategori puanı tamdır
    "H_CAT": 3.0,
    "W_CAT": {"kumar_h": 1.0, "oyun_h": 1.0, "sosyal_video_h": 0.7,
              "genel_web_h": 0.3, "iletisim_h": 0.15},
    "H_SLEEP": 1.0,
    "W_SLEEP": {"kumar_sleep_h": 1.2, "oyun_sleep_h": 1.2, "sosyal_video_sleep_h": 1.0,
                "genel_web_sleep_h": 0.6, "iletisim_sleep_h": 0.8, "dusuk_sleep_h": 0.3},
    "BASE_DAYS": 14,
    # geri dönüş: 3'ten sonra puan başlar, 15'te tam; nöbet: 30 dk'dan sonra başlar, 90 dk'da tam
    "LOOP": {"re_lo": 3, "re_hi": 15, "bout_lo": 30, "bout_hi": 90},
}
COMPONENTS = ["cat_pts", "sleep_pts", "dev_pts", "loop_pts"]
QUANTILES = (0.90, 0.97, 0.995)     # Dikkat, Yüksek, Kritik eşiklerinin yüzdelikleri
LEVELS = ["Normal", "Dikkat", "Yüksek", "Kritik"]

MAD_TO_SD = 1.4826      # ortanca mutlak sapmayı standart sapma ölçeğine çevirir
MAD_FLOOR_H = 0.5       # çok düzenli kullanıcıda küçük oynamalar sapma sayılmasın
Z_FULL = 3              # bu z-skorda sapma puanı tamdır


def _ramp(value, start, full):
    """start'ta 0, full'da 1 olan doğrusal rampa."""
    return np.clip((value - start) / (full - start), 0, 1)


def _weighted_hours(days, weights):
    return sum(weight * days[col] for col, weight in weights.items())


def _deviation_z(days, base_days):
    baseline = days.loc[days["day_idx"] < base_days].groupby("pid")["screen_h"]
    median = days["pid"].map(baseline.median())
    mad = days["pid"].map(baseline.agg(lambda h: (h - h.median()).abs().median()))
    z = (days["screen_h"] - median) / (MAD_TO_SD * mad + MAD_FLOOR_H)
    return z.where(days["day_idx"] >= base_days, 0.0)      # baz dönemde sapma puanı yok


def compute_components(days, p=DEFAULT):
    """Günlük ana tabloya bileşen puanlarını ve `risk` sütununu ekler."""
    days = days.sort_values(["pid", "date"]).reset_index(drop=True).copy()
    days["day_idx"] = days.groupby("pid").cumcount()
    pts, loop = p["PTS"], p["LOOP"]

    days["cat_pts"] = pts["cat"] * np.clip(_weighted_hours(days, p["W_CAT"]) / p["H_CAT"], 0, 1)
    days["sleep_pts"] = pts["sleep"] * np.clip(_weighted_hours(days, p["W_SLEEP"]) / p["H_SLEEP"], 0, 1)
    days["dev_pts"] = pts["dev"] * np.clip(_deviation_z(days, p["BASE_DAYS"]) / Z_FULL, 0, 1)
    reentry = _ramp(days["n_reentry"], loop["re_lo"], loop["re_hi"])
    long_bout = _ramp(days["longest_bout_min"], loop["bout_lo"], loop["bout_hi"])
    days["loop_pts"] = pts["loop"] * np.clip(0.5 * reentry + 0.5 * long_bout, 0, 1)

    days["risk"] = days[COMPONENTS].sum(axis=1) / sum(pts.values())
    return days


def thresholds(risk):
    """Yüzdelik tabanlı eşikler: [Dikkat, Yüksek, Kritik]."""
    return [float(q) for q in risk.quantile(list(QUANTILES))]


def add_levels(days, th=None):
    days = days.copy()
    th = th or thresholds(days["risk"])
    days["level"] = pd.cut(days["risk"], [-1, *th, 2], labels=LEVELS, right=False)
    days["flagged"] = days["level"] != "Normal"
    return days
