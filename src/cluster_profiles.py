"""11. adım: baz profil kümeleri. Her katılımcının ilk BASE_DAYS günündeki kullanım profili
KMeans ile gruplanır; amaç yeni kullanıcıya ilk iki haftanın sonunda bir başlangıç profili atamak.

Çıktı: profiles.csv (pid, cluster; yerelde kalır), reports/cluster_selection.csv,
       reports/cluster_profiles.csv, reports/fig_cluster_profiles.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from common import PROCESSED, REPORTS, load_analysis_days
from risk import DEFAULT, add_levels, compute_components

BASE_DAYS = DEFAULT["BASE_DAYS"]
MIN_BASE_DAYS = 10
N_CLUSTERS = 4
CANDIDATE_K = range(2, 7)
STABILITY_RUNS = 50
STABILITY_SHARE = 0.8
FEATURES = {"screen_h": "Ekran (sa/gün)", "sleep_use_h": "Uyku penc. kullanım (sa)",
            "oyun_h": "Oyun (sa)", "sosyal_video_h": "Sosyal/video (sa)", "iletisim_h": "İletişim (sa)",
            "genel_web_h": "Genel web (sa)", "n_sessions": "Oturum sayısı",
            "n_reentry": "Geri dönüş sayısı", "longest_bout_min": "En uzun nöbet (dk)"}


def baseline_profiles(scored):
    base = scored[scored["day_idx"] < BASE_DAYS]
    profiles = base.groupby("pid")[list(FEATURES)].mean()
    return profiles[base.groupby("pid").size() >= MIN_BASE_DAYS]


def compare_cluster_counts(X):
    """Her k için ayrışma (silhouette) ve alt örneklemede kararlılık (ARI)."""
    rng = np.random.default_rng(0)
    rows = []
    for k in CANDIDATE_K:
        full = KMeans(k, n_init=20, random_state=0).fit(X)
        agreement = []
        for _ in range(STABILITY_RUNS):
            subset = rng.choice(len(X), int(STABILITY_SHARE * len(X)), replace=False)
            refit = KMeans(k, n_init=5, random_state=int(rng.integers(1e9))).fit(X[subset])
            agreement.append(adjusted_rand_score(full.labels_[subset], refit.labels_))
        rows.append(dict(k=k, silhouette=silhouette_score(X, full.labels_),
                         kararlilik_ari=np.mean(agreement), en_kucuk_kume=np.bincount(full.labels_).min()))
    return pd.DataFrame(rows)


def assign_clusters(profiles, X):
    """Kümeler ortalama ekran süresine göre numaralanır (0 = en hafif)."""
    labels = pd.Series(KMeans(N_CLUSTERS, n_init=50, random_state=0).fit(X).labels_, index=profiles.index)
    lightest_first = profiles.groupby(labels)["screen_h"].mean().sort_values().index
    return labels.map({old: new for new, old in enumerate(lightest_first)}).rename("cluster")


def describe_clusters(profiles, clusters, scored):
    """Küme ortalamaları ve kümelerin baz dönemden sonraki risk durumu."""
    later = scored[scored["day_idx"] >= BASE_DAYS].groupby("pid").agg(
        sonraki_risk=("risk", "mean"), sonraki_uyari_payi=("flagged", "mean"))
    table = profiles.join(later).groupby(clusters).mean()
    table.insert(0, "n", clusters.value_counts())
    return table


def plot_centers(X, clusters, sizes):
    centers = pd.DataFrame(X, columns=list(FEATURES)).groupby(clusters.values).mean()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    image = ax.imshow(centers.values, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES.values(), rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(N_CLUSTERS))
    ax.set_yticklabels([f"Profil {i} (n={sizes[i]})" for i in range(N_CLUSTERS)])
    for (row, col), value in np.ndenumerate(centers.values):
        ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, label="z-skor (log ölçek)")
    ax.set_title(f"Baz profil kümeleri (ilk {BASE_DAYS} gün, KMeans k={N_CLUSTERS})")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_cluster_profiles.png", dpi=160)


def main():
    REPORTS.mkdir(exist_ok=True)
    scored = add_levels(compute_components(load_analysis_days()))
    profiles = baseline_profiles(scored)
    X = StandardScaler().fit_transform(np.log1p(profiles))      # sağa çarpık dağılımlar için log

    selection = compare_cluster_counts(X)
    selection.round(3).to_csv(REPORTS / "cluster_selection.csv", index=False)
    clusters = assign_clusters(profiles, X)
    clusters.to_frame().to_csv(PROCESSED / "profiles.csv")
    table = describe_clusters(profiles, clusters, scored)
    table.round(2).to_csv(REPORTS / "cluster_profiles.csv")
    plot_centers(X, clusters, table["n"])

    print(selection.round(3).to_string(index=False))
    print(table.round(2).T.to_string())


if __name__ == "__main__":
    main()
