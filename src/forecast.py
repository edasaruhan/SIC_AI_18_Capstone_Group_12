"""12. adım: ertesi gün risk tahmini. t gününe kadarki davranıştan, t+1 gününün
"Dikkat veya üstü" olup olmayacağı tahmin edilir. (Aynı günün skorunu tahmin etmek etiket
sızıntısı olurdu; burada bütün özellikler t gününde biter.)

Doğrulama: GroupKFold, katılımcıya göre - bir kişi ya eğitimde ya testte. Dikkat eşiği ve
karar eşiği her katta yalnızca eğitim katılımcılarından hesaplanır.
Çıktı: reports/forecast_metrics.csv, forecast_importance.csv, fig_forecast_pr.png,
       fig_forecast_importance.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import REPORTS, load_analysis_days
from risk import QUANTILES, compute_components

MIN_HISTORY_DAYS = 3
N_FOLDS, N_INNER_FOLDS = 5, 4
N_BOOTSTRAP = 500
TODAY_FEATURES = ["risk", "cat_pts", "sleep_pts", "dev_pts", "loop_pts", "screen_h", "sleep_use_h",
                  "oyun_h", "sosyal_video_h", "genel_web_h", "iletisim_h", "n_sessions",
                  "n_reentry", "longest_bout_min", "night_ratio"]
ROLLING_FEATURES = ["risk", "screen_h", "sleep_use_h", "cat_pts", "sleep_pts", "loop_pts"]
NOT_FEATURES = ["pid", "date", "risk_yarin"]

PERSISTENCE = "Dünkü seviye (kalıcılık)"
SCORE_BASELINES = {"Bugünkü risk skoru": "risk", "7 günlük ortalama risk": "risk_ort7"}
MODELS = {
    "Lojistik regresyon": lambda: make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
        class_weight="balanced", random_state=0),
}
IMPORTANCE_MODEL = "Gradient boosting"
METHODS = [PERSISTENCE, *SCORE_BASELINES, *MODELS]


def build_dataset(scored):
    """Her satır: t günündeki özellikler + t+1 gününün risk skoru (yalnızca ardışık günler)."""
    by_person = scored.groupby("pid")
    data = scored[["pid", "date", "day_idx"] + TODAY_FEATURES].copy()
    for col in ROLLING_FEATURES:
        for window in (3, 7):
            data[f"{col}_ort{window}"] = by_person[col].transform(
                lambda s: s.rolling(window, min_periods=1).mean())
    data["risk_std7"] = by_person["risk"].transform(lambda s: s.rolling(7, min_periods=2).std()).fillna(0)
    data["risk_max7"] = by_person["risk"].transform(lambda s: s.rolling(7, min_periods=1).max())
    data["risk_egilim"] = data["risk"] - data["risk_ort7"]
    data["risk_kisi_ort"] = by_person["risk"].transform(lambda s: s.expanding().mean())

    tomorrow = by_person[["date", "risk"]].shift(-1)
    data["risk_yarin"] = tomorrow["risk"]
    weekday = (data["date"] + pd.Timedelta(days=1)).dt.dayofweek
    data["yarin_haftasonu"] = (weekday >= 5).astype(int)
    data["yarin_cuma_ctesi"] = weekday.isin([4, 5]).astype(int)     # ertesi gece geç yatma olasılığı

    is_consecutive = (tomorrow["date"] - data["date"]).dt.days == 1
    data = data[is_consecutive & (data["day_idx"] >= MIN_HISTORY_DAYS)].reset_index(drop=True)
    return data, [c for c in data.columns if c not in NOT_FEATURES]


def best_f1_cutoff(y, score):
    precision, recall, cutoffs = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    return cutoffs[np.argmax(f1)]


def inner_cv_cutoff(make_model, X, y, groups):
    """Karar eşiği: eğitim katı içinde, yine kişiye göre bölünmüş iç doğrulamayla."""
    out_of_fold = np.zeros(len(y))
    for fit, held in GroupKFold(N_INNER_FOLDS).split(X, groups=groups):
        out_of_fold[held] = make_model().fit(X.iloc[fit], y[fit]).predict_proba(X.iloc[held])[:, 1]
    return best_f1_cutoff(y, out_of_fold)


def cross_validate(data, features, scored):
    n = len(data)
    result = dict(y=np.zeros(n, int), today_flagged=np.zeros(n, bool),
                  score={m: np.full(n, np.nan) for m in METHODS},
                  decision={m: np.zeros(n, bool) for m in METHODS})
    importances = []
    for train, test in GroupKFold(N_FOLDS).split(data, groups=data["pid"]):
        train_pids = data.loc[train, "pid"].unique()
        attention = scored.loc[scored["pid"].isin(train_pids), "risk"].quantile(QUANTILES[0])
        y_train = (data.loc[train, "risk_yarin"] >= attention).astype(int).values
        y_test = (data.loc[test, "risk_yarin"] >= attention).astype(int).values
        result["y"][test] = y_test
        result["today_flagged"][test] = (data.loc[test, "risk"] >= attention).values

        result["score"][PERSISTENCE][test] = result["today_flagged"][test].astype(float)
        result["decision"][PERSISTENCE][test] = result["today_flagged"][test]
        for name, col in SCORE_BASELINES.items():
            result["score"][name][test] = data.loc[test, col].values
            result["decision"][name][test] = (data.loc[test, col].values
                                              >= best_f1_cutoff(y_train, data.loc[train, col].values))
        X_train, X_test = data.loc[train, features], data.loc[test, features]
        for name, make_model in MODELS.items():
            cutoff = inner_cv_cutoff(make_model, X_train, y_train, data.loc[train, "pid"])
            model = make_model().fit(X_train, y_train)
            result["score"][name][test] = model.predict_proba(X_test)[:, 1]
            result["decision"][name][test] = result["score"][name][test] >= cutoff
            if name == IMPORTANCE_MODEL:
                permuted = permutation_importance(model, X_test, y_test, scoring="average_precision",
                                                  n_repeats=10, random_state=0)
                importances.append(pd.Series(permuted.importances_mean, index=features))
    result["importance"] = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return result


def metric_rows(result, mask, subset):
    y = result["y"][mask]
    rows = []
    for name in METHODS:
        score, decision = result["score"][name][mask], result["decision"][name][mask]
        rows.append(dict(alt_kume=subset, model=name, n=int(mask.sum()), pozitif_oran=y.mean(),
                         pr_auc=average_precision_score(y, score),
                         roc_auc=roc_auc_score(y, score) if len(set(score)) > 1 else np.nan,
                         f1=f1_score(y, decision, zero_division=0),
                         kesinlik=precision_score(y, decision, zero_division=0),
                         duyarlilik=recall_score(y, decision, zero_division=0)))
    return rows


def bootstrap_pr_auc(data, result):
    """Katılımcıları yeniden örnekleyerek her yöntemin PR-AUC dağılımı."""
    rng = np.random.default_rng(0)
    rows_of = data.groupby("pid").indices
    pids = list(rows_of)
    methods = [m for m in METHODS if m != PERSISTENCE]
    draws = {m: [] for m in methods}
    for _ in range(N_BOOTSTRAP):
        rows = np.concatenate([rows_of[pid] for pid in rng.choice(pids, len(pids))])
        for m in methods:
            draws[m].append(average_precision_score(result["y"][rows], result["score"][m][rows]))
    return pd.DataFrame(draws)


def interval(values):
    return f"{np.percentile(values, 2.5):.3f} - {np.percentile(values, 97.5):.3f}"


def plot_precision_recall(result):
    y = result["y"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for name in METHODS[1:]:
        precision, recall, _ = precision_recall_curve(y, result["score"][name])
        ax.plot(recall, precision, label=f"{name} (PR-AUC {average_precision_score(y, result['score'][name]):.2f})")
    ax.axhline(y.mean(), ls=":", c="gray", label=f"Rastgele ({y.mean():.2f})")
    ax.set_xlabel("Duyarlılık (recall)")
    ax.set_ylabel("Kesinlik (precision)")
    ax.set_title(f"Ertesi gün 'Dikkat+' tahmini ({N_FOLDS} kat, kişiye göre)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_forecast_pr.png", dpi=160)


def plot_importance(importance):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    importance.head(12)[::-1].plot.barh(ax=ax, color="#3b6ea5")
    ax.set_xlabel("Karıştırılınca PR-AUC düşüşü")
    ax.set_title(f"Özellik önemi ({IMPORTANCE_MODEL.lower()})")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_forecast_importance.png", dpi=160)


def main():
    REPORTS.mkdir(exist_ok=True)
    scored = compute_components(load_analysis_days())
    data, features = build_dataset(scored)
    print("Tahmin örneği (katılımcı-gün):", len(data), "| katılımcı:", data["pid"].nunique(),
          "| özellik:", len(features))

    result = cross_validate(data, features, scored)
    everything = np.ones(len(data), bool)
    metrics = pd.DataFrame(metric_rows(result, everything, "Tüm günler")
                           + metric_rows(result, ~result["today_flagged"], "Yeni başlangıç (bugün Normal)"))
    metrics.round(3).to_csv(REPORTS / "forecast_metrics.csv", index=False)
    result["importance"].round(4).rename("pr_auc_dususu").to_csv(REPORTS / "forecast_importance.csv")
    plot_precision_recall(result)
    plot_importance(result["importance"])

    print("\n" + metrics.round(3).to_string(index=False))
    boot = bootstrap_pr_auc(data, result)
    print("\nPR-AUC %95 güven aralığı (katılımcı bootstrap):")
    for name in boot:
        print(f"  {name}: {interval(boot[name])}")
    for baseline in SCORE_BASELINES:
        print(f"  Fark ({IMPORTANCE_MODEL} - {baseline}): {interval(boot[IMPORTANCE_MODEL] - boot[baseline])}")
    print("\nEn önemli 10 özellik (karıştırılınca PR-AUC düşüşü):")
    print(result["importance"].head(10).round(4).to_string())


if __name__ == "__main__":
    main()
