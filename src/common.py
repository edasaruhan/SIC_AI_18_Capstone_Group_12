"""Bütün betiklerin paylaştığı yollar, sabitler ve yardımcı fonksiyonlar."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("mobile_phone_use/data")
PINFO_PATH = Path("mobile_phone_use/pinfo.csv")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")
CONFIG = Path("config")

NON_APP_PREFIXES = ("+", "=", "-")   # + başlatıcı, = klavye, - Android sistemi
STUDY_APP = "org.pielot.sns2"        # veri toplama uygulamasının kendisi
STUDY_PREFIX = "org.pielot"          # aynı geliştiricinin bütün paketleri
RISKY_GROUPS = ["kumar", "oyun", "sosyal_video"]
RISK_GROUPS = RISKY_GROUPS + ["genel_web", "iletisim", "dusuk"]

WINDOW_ANCHOR_H = 18                 # uyku penceresi dakikaları gecenin 18:00'inden ölçülür
MIN_WINDOW_NIGHTS = 7                # kişisel pencereye güvenmek için gereken gece sayısı
MIN_WINDOW_COVER = 0.5               # ... ve uyku boşluğu bulunan gün payı

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def raw_files():
    """(pid, dosya yolu) çiftleri. pid, dosya adına göre sıradan gelir: P000, P001, ..."""
    for i, path in enumerate(sorted(RAW_DIR.glob("*.csv"))):
        yield f"P{i:03d}", path


def is_real_app(app):
    """Kullanıcının açtığı gerçek uygulama mı? (başlatıcı, klavye, sistem ve çalışma uygulaması değil)"""
    return ~app.str.startswith(NON_APP_PREFIXES) & (app != STUDY_APP)


def load_ok_participants():
    return pd.read_csv(PROCESSED / "participants_ok.csv")


def load_valid_sessions():
    """Kalite filtresinden geçen katılımcıların geçerli oturumları (yerel bitiş zamanıyla)."""
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    ok_pids = load_ok_participants()["pid"]
    sessions = sessions[sessions["valid"] & sessions["pid"].isin(ok_pids)].copy()
    sessions["end_local"] = sessions["start"] + pd.to_timedelta(sessions["duration_s"], unit="s")
    return sessions


def load_sleep_windows():
    """Kişisel uyku pencereleri. Penceresi güvenilmez olanlara güvenilir grubun ortancası atanır."""
    win = pd.read_csv(PROCESSED / "sleep_windows.csv")[["pid", "n_nights", "bed_min", "wake_min", "night_cover"]]
    win["reliable"] = (win["night_cover"] >= MIN_WINDOW_COVER) & (win["n_nights"] >= MIN_WINDOW_NIGHTS)
    for col in ("bed_min", "wake_min"):
        win[col] = np.where(win["reliable"], win[col], win.loc[win["reliable"], col].median())
    return win[["pid", "bed_min", "wake_min", "reliable"]]


def window_overlap_s(start, end, bed_min, wake_min, day_shift):
    """[start, end) aralığının, (başlangıç günü + day_shift) akşamında başlayan uyku penceresiyle
    çakıştığı saniye. Bir olay hem önceki gecenin (day_shift=-1) hem o gecenin (0) penceresine değebilir."""
    anchor = start.dt.normalize() + pd.Timedelta(days=day_shift, hours=WINDOW_ANCHOR_H)
    window_start = anchor + pd.to_timedelta(bed_min, unit="m")
    window_end = anchor + pd.to_timedelta(wake_min, unit="m")
    lo = np.maximum(start.values, window_start.values)
    hi = np.minimum(end.values, window_end.values)
    return np.clip((hi - lo) / np.timedelta64(1, "s"), 0, None)


def load_analysis_days():
    """Günlük ana tablo; gece ekranı sürekli açık görünen (şüpheli) katılımcılar hariç."""
    master = pd.read_parquet(PROCESSED / "master_daily.parquet")
    return master[~master["night_suspect"]].copy()
