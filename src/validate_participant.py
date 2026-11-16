"""9. adım: dış geçerlilik (katılımcı düzeyi). Risk skoru ve bileşenleri, katılımcının
öz-bildirim ölçümleriyle (ruh hali ortalamaları, PHQ-8, BPS, Big5) ilişkili mi?

Çıktı: reports/validity_participant.csv (yalnızca toplu istatistik)
"""
import numpy as np
import pandas as pd
from scipy import stats
from common import PINFO_PATH, PROCESSED, REPORTS, load_analysis_days
from risk import add_levels, compute_components

PREDICTORS = ["risk_ort", "dikkat_gun_payi", "kategori", "uyku", "sapma", "dongu",
              "ekran_saat", "uyku_kullanim_saat"]
OUTCOMES = {"valence_mean": "Ruh hali (yüksek=olumlu)", "tired_awake_mean": "Uyanıklık (yüksek=uyanık)",
            "tense_calm_mean": "Sakinlik (yüksek=sakin)", "bored_frac": "Sıkılma payı",
            "PHQ8": "PHQ-8 depresyon", "BPSScore": "Sıkılma yatkınlığı (BPS)",
            "Big5Conscientiousness": "Sorumluluk", "Big5Neuroticism": "Nevrotiklik", "age": "Yaş"}
COVARIATES = ["age", "gender"]


def participant_table():
    scored = add_levels(compute_components(load_analysis_days()))
    per_person = scored.groupby("pid").agg(
        risk_ort=("risk", "mean"), dikkat_gun_payi=("flagged", "mean"),
        kategori=("cat_pts", "mean"), uyku=("sleep_pts", "mean"),
        sapma=("dev_pts", "mean"), dongu=("loop_pts", "mean"),
        ekran_saat=("screen_h", "mean"), uyku_kullanim_saat=("sleep_use_h", "mean")).reset_index()
    pid_map = pd.read_csv(PROCESSED / "pid_map.csv")
    pinfo = pd.read_csv(PINFO_PATH).rename(columns={"uuid": "file_stem"})
    return per_person.merge(pid_map, on="pid").merge(pinfo, on="file_stem")


def partial_spearman(x, y, covariates):
    """Ortak değişkenler kontrol edilmiş Spearman: sıralar üzerinde artıkların korelasyonu."""
    design = np.column_stack([np.ones(len(x))] + [stats.rankdata(c) for c in covariates])

    def residual(values):
        ranks = stats.rankdata(values)
        return ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]

    rho = np.corrcoef(residual(x), residual(y))[0, 1]
    dof = len(x) - 2 - len(covariates)
    t = rho * np.sqrt(dof / max(1e-12, 1 - rho ** 2))
    return rho, 2 * stats.t.sf(abs(t), dof)


def benjamini_hochberg(p_values):
    """Çoklu karşılaştırma için düzeltilmiş p (q) değerleri."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate(scaled[::-1])[::-1]
    return np.clip(q, 0, 1)


def correlate(data, predictor, outcome):
    if outcome in COVARIATES:      # yaşın kendisi için kısmi korelasyon tanımsız
        sample = data[[predictor, outcome]].dropna()
        partial = (np.nan, np.nan)
    else:
        sample = data[[predictor, outcome] + COVARIATES].dropna()
        partial = partial_spearman(sample[predictor].values, sample[outcome].values,
                                   [sample[c].values for c in COVARIATES])
    rho, p = stats.spearmanr(sample[predictor], sample[outcome])
    return dict(olcum=OUTCOMES[outcome], degisken=predictor, n=len(sample), rho=rho, p=p,
                rho_kismi=partial[0], p_kismi=partial[1])


def main():
    data = participant_table()
    results = pd.DataFrame([correlate(data, x, y) for y in OUTCOMES for x in PREDICTORS])
    results["q_bh"] = benjamini_hochberg(results["p"])
    REPORTS.mkdir(exist_ok=True)
    results.round(4).to_csv(REPORTS / "validity_participant.csv", index=False)

    print("Katılımcı:", len(data))
    print(results.round(3).to_string(index=False))
    print("\nDüzeltmeden sonra anlamlı (q_bh < 0.05):")
    print(results[results["q_bh"] < 0.05].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
