"""8. adım: risk skorunu hesaplar, eşikleri veriden belirler ve sağlamlığını sınar.

Çıktı: risk_daily.parquet, config/risk_thresholds.json, reports/calibration.txt
"""
import copy
import json
import numpy as np
import pandas as pd
from common import CONFIG, PROCESSED, REPORTS, load_analysis_days
from risk import COMPONENTS, DEFAULT, LEVELS, QUANTILES, add_levels, compute_components, thresholds

N_PERTURBATIONS = 30
PERTURB_RANGE = (0.75, 1.25)
TOP_SHARE = 0.10


def top_days(risk):
    return set(risk.nlargest(int(TOP_SHARE * len(risk))).index)


def agreement(reference, other):
    """İki skorun uyumu: Spearman ve en riskli günlerin Jaccard örtüşmesi."""
    a, b = top_days(reference), top_days(other)
    return reference.corr(other, method="spearman"), len(a & b) / len(a | b)


def perturbed(params, rng):
    """Bütün ağırlık ve ölçek parametrelerini rastgele ±%25 oynatır."""
    p = copy.deepcopy(params)

    def scale():
        return rng.uniform(*PERTURB_RANGE)

    for key in p["PTS"]:
        p["PTS"][key] *= scale()
    for key in ("H_CAT", "H_SLEEP"):
        p[key] *= scale()
    for group in ("W_CAT", "W_SLEEP"):
        for key in p[group]:
            p[group][key] *= scale()
    for key in ("re_hi", "bout_hi"):
        p["LOOP"][key] *= scale()
    return p


def warning_burden(scored):
    share = scored.groupby("pid")["flagged"].mean()
    critical = scored.loc[scored["level"] == "Kritik", "pid"].nunique() / scored["pid"].nunique()
    return [
        "\nKatılımcı başına Dikkat veya üstü gün payı:",
        share.describe(percentiles=[0.5, 0.9]).round(3).to_string(),
        f"Hiç uyarı almayan katılımcı payı: {(share == 0).mean():.3f}",
        f"En az bir Kritik günü olan katılımcı payı: {critical:.3f}",
    ]


def sensitivity(days, risk):
    rng = np.random.default_rng(0)
    runs = [agreement(risk, compute_components(days, perturbed(DEFAULT, rng))["risk"])
            for _ in range(N_PERTURBATIONS)]
    table = pd.DataFrame(runs, columns=["spearman", "top10_ortusme"])
    return [f"\nDuyarlılık ({N_PERTURBATIONS} rastgele parametre seti, ±%25):",
            table.describe().loc[["mean", "min"]].round(3).to_string()]


def ablation(scored):
    rows = []
    for dropped in COMPONENTS:
        rest = scored[[c for c in COMPONENTS if c != dropped]].sum(axis=1)
        rows.append((dropped, *agreement(scored["risk"], rest)))
    table = pd.DataFrame(rows, columns=["cikarilan", "spearman", "top10_ortusme"])
    return ["\nAblasyon (bileşen çıkarılınca tam skorla uyum):", table.round(3).to_string(index=False)]


def main():
    days = load_analysis_days()
    scored = compute_components(days)
    th = thresholds(scored["risk"])
    scored = add_levels(scored, th)
    scored.to_parquet(PROCESSED / "risk_daily.parquet")
    with open(CONFIG / "risk_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"quantiles": QUANTILES, "levels": LEVELS[1:], "thresholds": [round(t, 4) for t in th],
                   "params": DEFAULT, "n_participant_days": len(scored),
                   "n_participants": int(scored["pid"].nunique())}, f, ensure_ascii=False, indent=2)

    lines = [
        f"Katılımcı-gün: {len(scored)} | katılımcı: {scored['pid'].nunique()}",
        "Eşikler (risk, 0-1): " + " | ".join(f"{name} >= {t:.3f}" for name, t in zip(LEVELS[1:], th)),
        *warning_burden(scored),
        "\nSeviyeye göre bileşen ortalamaları (puan):",
        scored.groupby("level", observed=True)[COMPONENTS].mean().round(1).to_string(),
        "\nBileşenler arası Spearman korelasyonu:",
        scored[COMPONENTS].corr(method="spearman").round(2).to_string(),
        *sensitivity(days, scored["risk"]),
        *ablation(scored),
    ]
    report = "\n".join(lines)
    print(report)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "calibration.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
