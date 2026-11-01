from pathlib import Path
import pandas as pd

OUT = Path("data/processed")
summ = pd.read_csv(OUT / "participant_summary.csv")

# kimlik eşleme (build_sessions.py ile aynı sıra, yerelde kalır)
files = sorted(Path("mobile_phone_use/data").glob("*.csv"))
pd.DataFrame({"pid": [f"P{i:03d}" for i in range(len(files))],
              "file_stem": [f.stem for f in files]}).to_csv(OUT / "pid_map.csv", index=False)

summ["ok_days"] = summ["n_days"] >= 14
summ["ok_invalid"] = summ["invalid_share"] < 0.05
summ["extreme_screen"] = summ["screen_h_per_day"] > 16
summ["unlock_reliable"] = summ["unlocked_share"].between(0.05, 0.95)
summ["keep"] = summ[["ok_days", "ok_invalid"]].all(axis=1)
summ["keep"] = summ["keep"] & ~summ["pid"].isin(["P208"])

print("Toplam katılımcı:", len(summ))
print("Yetersiz gün (<14):", int((~summ["ok_days"]).sum()))
print("Geçersiz oturum payı yüksek:", int((~summ["ok_invalid"]).sum()))
print("Günde 16 saatten fazla ekran (işaretli, dışlanmadı):", int(summ["extreme_screen"].sum()))
print(summ.loc[summ["extreme_screen"],
               ["pid", "n_days", "invalid_share", "screen_h_per_day", "unlocked_share"]])
print("Kalan katılımcı:", int(summ["keep"].sum()))
print("Kalanlar içinde kilit bilgisi güvenilmez olan:",
      int((summ["keep"] & ~summ["unlock_reliable"]).sum()))

pd.set_option("display.width", 200)
print("\nKalanlarda günlük ekran saati:")
print(summ.loc[summ["keep"], "screen_h_per_day"].describe().round(2))

summ[summ["keep"]].to_csv(OUT / "participants_ok.csv", index=False)