"""2. adım: kalite filtresi. Yeterli günü olan ve oturumları tutarlı katılımcıları seçer.

Çıktı: participants_ok.csv
"""
import pandas as pd
from common import PROCESSED

MIN_DAYS = 14
MAX_INVALID_SHARE = 0.05
EXTREME_SCREEN_H = 16                 # yalnızca işaretlenir, tek başına dışlama nedeni değil
UNLOCK_SHARE_RANGE = (0.05, 0.95)     # dışındaysa kilit açma kaydı güvenilmez sayılır
# Elle incelenip dışlananlar. P208: ekran günde ortalama 18 saatten fazla açık görünüyor.
MANUAL_EXCLUDE = ["P208"]


def main():
    summary = pd.read_csv(PROCESSED / "participant_summary.csv")
    summary["ok_days"] = summary["n_days"] >= MIN_DAYS
    summary["ok_invalid"] = summary["invalid_share"] < MAX_INVALID_SHARE
    summary["extreme_screen"] = summary["screen_h_per_day"] > EXTREME_SCREEN_H
    summary["unlock_reliable"] = summary["unlocked_share"].between(*UNLOCK_SHARE_RANGE)
    summary["keep"] = summary["ok_days"] & summary["ok_invalid"] & ~summary["pid"].isin(MANUAL_EXCLUDE)
    kept = summary[summary["keep"]]
    kept.to_csv(PROCESSED / "participants_ok.csv", index=False)

    print("Toplam katılımcı:", len(summary))
    print(f"Yetersiz gün (<{MIN_DAYS}):", int((~summary["ok_days"]).sum()))
    print("Geçersiz oturum payı yüksek:", int((~summary["ok_invalid"]).sum()))
    print(f"Günde {EXTREME_SCREEN_H} saatten fazla ekran (işaretli):", int(summary["extreme_screen"].sum()))
    print("Elle dışlanan:", MANUAL_EXCLUDE)
    print("Kalan katılımcı:", len(kept), "| kilit bilgisi güvenilmez:", int((~kept["unlock_reliable"]).sum()))
    print("\nKalanlarda günlük ekran saati:")
    print(kept["screen_h_per_day"].describe().round(2))


if __name__ == "__main__":
    main()
