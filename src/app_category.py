"""5b. adım: uygulama paketlerini kategoriye ve risk grubuna eşler.

Sıra: config/app_categories.csv'de tam eşleşme → ön ek kuralı → "bilinmeyen" (genel_web ağırlığı).
Çıktı: app_counts_labeled.csv. Eşlenmemiş en sık uygulamalar ekrana yazılır; bunları
config/app_categories.csv'ye ekleyip betiği yeniden çalıştırın.
"""
import pandas as pd
from common import CONFIG, PROCESSED, STUDY_PREFIX

GAME_PREFIXES = ("com.king.", "com.supercell.", "com.midasplayer", "com.zynga.",
                 "com.ea.game", "com.etermax.", "com.nianticlabs.")
N_UNMAPPED_SHOWN = 25


def classify(package, mapping):
    """(category, risk_group, exclude, how) döndürür."""
    if package in mapping.index:
        row = mapping.loc[package]
        return row["category"], row["risk_group"], int(row["exclude"]), "tam"
    if package.startswith(STUDY_PREFIX):
        return "calisma_uygulamasi", "dusuk", 1, "kural"
    if package.startswith(GAME_PREFIXES):
        return "oyun", "oyun", 0, "kural"
    return "bilinmeyen", "genel_web", 0, "yok"


def main():
    counts = pd.read_csv(PROCESSED / "app_counts.csv", index_col=0)
    mapping = pd.read_csv(CONFIG / "app_categories.csv").set_index("package")
    labels = pd.DataFrame([classify(pkg, mapping) for pkg in counts.index], index=counts.index,
                          columns=["category", "risk_group", "exclude", "how"])
    labeled = counts.join(labels)
    labeled.to_csv(PROCESSED / "app_counts_labeled.csv")

    total = labeled["events"].sum()
    print("Olay payı, eşleme yöntemine göre:")
    print((labeled.groupby("how")["events"].sum() / total).round(3))
    print("\nOlay payı, risk grubuna göre:")
    print((labeled.groupby("risk_group")["events"].sum() / total).round(3))
    print(f"\nEşlenmemiş en sık {N_UNMAPPED_SHOWN} uygulama:")
    unmapped = labeled[labeled["how"] == "yok"].sort_values("events", ascending=False)
    print(unmapped.head(N_UNMAPPED_SHOWN)[["events", "users"]].to_string())


if __name__ == "__main__":
    main()
