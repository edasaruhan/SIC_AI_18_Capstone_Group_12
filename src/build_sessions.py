from pathlib import Path
import pandas as pd
from sessions import load_participant, sessionize

DATA = Path("mobile_phone_use/data")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

all_sessions, summary = [], []
for i, f in enumerate(sorted(DATA.glob("*.csv"))):
    pid = f"P{i:03d}"
    try:
        df = load_participant(f)
        s = sessionize(df)
    except Exception as e:
        print(pid, "hata:", type(e).__name__, e)
        continue
    if s.empty:
        summary.append({"pid": pid, "n_sessions": 0})
        continue
    s.insert(0, "pid", pid)
    all_sessions.append(s)
    sv = s[s["valid"]]
    days = max(sv["date"].nunique(), 1)
    summary.append({
        "pid": pid,
        "n_days": days,
        "n_sessions": len(s),
        "invalid_share": 1 - s["valid"].mean(),
        "n_negative": int((s["duration_s"] <= 0).sum()),
        "n_too_long": int((s["duration_s"] > 12 * 3600).sum()),
        "unlocked_share": sv["unlocked"].mean(),
        "screen_h_per_day": sv["duration_s"].sum() / 3600 / days,
        "median_session_s": sv["duration_s"].median(),
    })

sessions = pd.concat(all_sessions, ignore_index=True)
summ = pd.DataFrame(summary)
sessions.to_parquet(OUT / "sessions.parquet")
summ.to_csv(OUT / "participant_summary.csv", index=False)

print("Katılımcı:", len(summ), "| oturum:", len(sessions),
      "| geçersiz oturum payı:", round(1 - sessions["valid"].mean(), 4))
print(summ[["n_days", "n_sessions", "invalid_share", "unlocked_share",
            "screen_h_per_day", "median_session_s"]].describe().round(2))
print("\nOturumu negatif süreli olan katılımcı sayısı:", int((summ["n_negative"] > 0).sum()))
print("Günde 16 saatten fazla ekran açık görünen katılımcı sayısı:",
      int((summ["screen_h_per_day"] > 16).sum()))
print("unlocked_share < 0.05 olan katılımcı sayısı:", int((summ["unlocked_share"] < 0.05).sum()))
print("unlocked_share > 0.95 olan katılımcı sayısı:", int((summ["unlocked_share"] > 0.95).sum()))