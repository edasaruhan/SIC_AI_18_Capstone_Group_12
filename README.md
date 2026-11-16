# UNHOOK

Üniversite öğrencileri için, cihaz üzerinde çalışan, kademeli bant genişliği kısıtlaması ve
kullanıcı onayı (Human-in-the-Loop) içeren dijital denge sistemi. SIC AI Capstone, Grup 12.

Bu depo **Aşama 1 (veri ve baz model)** kodunu içerir: Pielot ve ark. "Mobile Phone Use" veri
setinden (CRAWDAD, İspanya, Nisan–Temmuz 2016) günlük özellikler, risk skoru, baz profiller ve
ertesi gün risk tahmini.

> Ham veri ve katılımcı düzeyindeki tablolar depoya **konmaz** (`.gitignore`). Depoda yalnızca
> kod, kategori eşlemesi ve toplu sonuçlar (`reports/`) bulunur.

## Kurulum

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Ham veriyi `mobile_phone_use/` altına açın (`data/*.csv`, `pinfo.csv`, `Sensors.xlsx`).
Betikler proje kökünden `python src/<betik>.py` ile çalışır.

## Çalıştırma sırası

| # | Betik | Ne yapar | Çıktı (`data/processed/`) |
|---|---|---|---|
| 1 | `build_sessions.py` | Ekran olaylarından oturum çıkarır | `sessions.parquet`, `pid_map.csv` |
| 2 | `quality.py` | Kalite filtresi (≥14 gün, geçersiz oturum <%5) | `participants_ok.csv` |
| 3 | `daily_features.py` | Gece yarısında bölünmüş günlük ekran özellikleri | `daily_features.parquet` |
| 4 | `sleep_window.py`, `sleep_overlap.py`, `night_activity.py` | Kişisel uyku penceresi; "gece ekranı hep açık" şüphelileri | `sleep_windows.csv`, `night_activity.csv` |
| 5 | `app_counts.py`, `app_category.py`, `build_app_events.py` | Uygulama → kategori, olay başına süre | `app_events.parquet` |
| 6 | `loop_features.py` | Riskli kategoride nöbet ve geri dönüş | `loop_daily.parquet` |
| 7 | `daily_master.py` | Günlük ana tablo | `master_daily.parquet` |
| 8 | `calibrate.py` | Risk skoru, eşikler, duyarlılık, ablasyon | `risk_daily.parquet`, `config/risk_thresholds.json` |
| 9 | `validate_participant.py` | Dış geçerlilik (katılımcı düzeyi) | `reports/validity_participant.csv` |
| 10 | `extract_esm.py`, `validate_daily.py` | Dış geçerlilik (gün düzeyi, kişi-içi) | `esm.parquet`, `reports/validity_daily.csv` |
| 11 | `cluster_profiles.py` | Baz profil kümeleri | `profiles.csv`, `reports/cluster_*.csv` |
| 12 | `forecast.py` | Ertesi gün risk tahmini | `reports/forecast_*.csv` |

`common.py` (yollar, ortak sabitler, yardımcılar) ve `risk.py` (risk skorunun tanımı) betik değil,
diğerlerinin kullandığı modüllerdir. Hepsini sırayla çalıştırmak için (PowerShell):

```
"build_sessions","quality","daily_features","sleep_window","sleep_overlap","night_activity","app_counts","app_category","build_app_events","loop_features","daily_master","calibrate","validate_participant","extract_esm","validate_daily","cluster_profiles","forecast" | % { python "src/$_.py"; if ($LASTEXITCODE) { throw "$_ hata verdi" } }
```

## Risk skoru

Günlük, 0–1 arası, toplamsal. Dört bileşen (toplam 80 puan; "planlı görev gecikmesi" bileşeni
bu veri setinde ölçülemediği için yok, uygulamada 100 puana tamamlanacak):

| Bileşen | Puan | Tanım |
|---|---|---|
| Kategori | 20 | Ağırlıklı süre: kumar 1,0 · oyun 1,0 · sosyal/video 0,7 · genel web 0,3 · iletişim 0,15; 3 ağırlıklı saatte tam puan |
| Uyku penceresi | 25 | Kişinin kendi uyku penceresine denk gelen ağırlıklı kullanım; 1 saatte tam puan |
| Kişisel sapma | 20 | İlk 14 günün ortancasına göre sağlam z-skor; ilk 14 gün 0 |
| Döngü | 15 | Riskli kategoriye 15 dk içinde geri dönüş sayısı (3→15) ve en uzun kesintisiz riskli nöbet (30→90 dk) |

Eşikler veriden, yüzdelikle (90 / 97 / 99,5) belirlenir: **Dikkat ≥ 0,340 · Yüksek ≥ 0,542 · Kritik ≥ 0,737**.
Parametreler `src/risk.py` içinde, güncel eşikler `config/risk_thresholds.json` içindedir.

## Sonuçlar (274 katılımcı, 7.394 katılımcı-gün)

**Kalibrasyon.** Katılımcıların %48'i hiç uyarı almıyor; ortanca katılımcıda günlerin %3'ü,
en üst %10'da günlerin %37'sinden fazlası Dikkat veya üstü. Parametreler ±%25 oynatıldığında
sıralama korunuyor (Spearman ort. 0,997, en düşük 0,992; en riskli %10'un örtüşmesi ort. 0,85).

**Dış geçerlilik (katılımcı düzeyi).** Ortalama risk ile öz-bildirim ruh hali arasındaki
ilişkiler beklenen yönde ama zayıf (|ρ| ≤ 0,14) ve çoklu karşılaştırma düzeltmesinden sonra
anlamlı değil; yaş ve cinsiyet kontrol edilince daha da küçülüyor. Düzeltmeden sonra kalan tek
ilişki: uyku penceresi bileşeni ile sıkılma yatkınlığı (BPS) ρ = 0,32 (n = 78). PHQ-8 için
n = 54, sonuç keşifsel (ρ = 0,18, anlamsız). Risk yaşla belirgin biçimde azalıyor (ρ = −0,28).
Yani skor şu an bir **kullanım örüntüsü göstergesi**; ruh sağlığı sonucu olarak doğrulanmış değil.

**Dış geçerlilik (gün düzeyi, kişi-içi).** 274 katılımcı, 6.622 ruh hali bildirimli gün. Önceki
günün risk skoru ertesi günün ruh halini, uyanıklığını, sakinliğini veya sıkılmasını öngörmüyor
(kişi-içi |r| ≤ 0,03). 30 testten 2'si düzeltmesiz %95 aralıkta anlamlı; bu sayı şansla beklenene
yakın. Beklenen yöndeki tek bulgu: uyku penceresinde fazladan her saat kullanım, ertesi sabah
uyanıklığı −2…2 ölçeğinde 0,09 puan düşürüyor (%95 GA −0,18 – −0,01; kişi-içi r = −0,04).

**Baz profiller (ilk 14 gün, KMeans k = 4).** Kümeler keskin ayrılmıyor (silhouette 0,22,
alt örnekleme kararlılığı ARI 0,77); kullanım bir süreklilik, profiller kaba bir özet.

| Profil | n | Ekran sa/gün | Öne çıkan | Sonraki dönemde uyarılı gün payı |
|---|---|---|---|---|
| 0 Hafif | 96 | 2,1 | Riskli kategori neredeyse yok | %2 |
| 1 Sosyal, kısa ve sık | 90 | 3,2 | Sosyal/video 0,6 sa, sık oturum | %7 |
| 2 Uzun oturum, gece | 49 | 6,2 | Web 1,5 sa, oyun 0,4 sa, uyku penceresinde 0,4 sa | %23 |
| 3 Yoğun sosyal döngü | 39 | 6,6 | Sosyal/video 1,5 sa, günde 12 geri dönüş | %46 |

**Ertesi gün tahmini** (hedef: yarın Dikkat veya üstü; taban oran %11; 5 kat, kişiye göre).

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Dünkü seviye (kalıcılık) | 0,28 | 0,70 | 0,48 |
| Bugünkü risk skoru | 0,44 | 0,85 | 0,49 |
| 7 günlük ortalama risk | 0,54 | 0,89 | 0,56 |
| Lojistik regresyon | 0,55 | 0,89 | 0,55 |
| Gradient boosting | 0,54 | 0,89 | 0,56 |

Makine öğrenmesi modelleri bugünkü skordan anlamlı biçimde iyi (PR-AUC farkı %95 GA 0,07–0,13),
ama **7 günlük kişisel ortalamadan iyi değil** (fark GA −0,03 – +0,02). En önemli özellikler
kişinin o güne kadarki ortalama riski ve son 7 günün ortalaması. "Bugün Normal, yarın uyarılı"
durumlarında (taban %6,5) en iyi F1 0,40.

## Sınırlılıklar

- **Örneklem öğrenci değil.** Yaş ortancası 38 (18–66); 18–25 yaş yalnızca 47 kişi. Bu grupta
  ortalama risk ve uyarılı gün payı en yüksek (%16,1; 46+ yaşta %7,4). Eşikler öğrenci verisiyle
  yeniden kalibre edilmeli.
- Veri 2016 İspanya; uygulama ekosistemi (ör. kısa video) bugünkünden farklı.
- Eşikler yüzdelik tabanlı: "Dikkat" tanım gereği günlerin %10'u. Klinik bir kesme noktası değil.
- Etiket, özelliklerle aynı sinyalden türetiliyor. Tahmin başarısı "davranış sürekliliğini"
  ölçer, bağımsız bir zarar ölçütünü değil.
- Kategori ve döngü bileşenleri ilişkili (ρ = 0,68); döngü bileşeni çıkarılınca sıralama
  neredeyse değişmiyor (Spearman 0,99).
- Uygulama olaylarının %10'u eşlenmemiş (düşük ağırlıkla sayılıyor). Uyku penceresi ekran
  kapalı aralığından tahmin ediliyor, gerçek uyku ölçümü değil. 46 katılımcı gece ekranı sürekli
  açık göründüğü için ana analiz dışında.

## Uygulama tarafı: tasarım kararları

- İlk aşama yalnızca Android. Trafik cihazdaki yerel VPN (`VpnService`) üzerinden izlenir; yalnızca
  **alan adı** okunur (DNS/SNI), sayfa içeriği okunmaz, veri cihazdan çıkmaz.
- Alan adı cihazda kategoriye çevrilir. Bilinmeyen siteler için sıra: önbellek → paketlenmiş
  liste → alan adı kuralları → düşük ağırlıklı "bilinmeyen" → kullanıcıya bir kez sormak.
- Alan adı görünürlüğü Android'de denendi; alan adları görünüyor. ECH ve şifreli DNS yaygınlaştıkça
  görünürlük azalabilir, bu yüzden uygulama düzeyi kullanım verisi (`UsageStats`) ikinci kaynak.
- Müdahale kademelidir ve kullanıcı onayı içerir; eşikler `config/risk_thresholds.json` içinden okunur.

## Durum

Aşama 1 tamamlandı (bu depo). Sırada: `UsageStats` geçmiş saklama süresinin ve Health Connect uyku
verisinin denenmesi, eşiklerin öğrenci verisiyle yeniden kalibrasyonu, müdahale katmanı (Aşama 2).

## Kaynaklar

- Veri seti: Pielot ve ark., "Mobile Phone Use" (CRAWDAD). Tam künye veri setiyle gelen
  `Mobile Phone Use Log Data Set.pdf` dosyasındaki atıf talimatına göre yazılmalı.
- King, Billieux ve Delfabbro (2024), "Red Box, Green Box" çerçevesi.
- Stevens ve ark. (2026), *Addiction*, doi:10.1111/add.70192.
- WHO ICD-11: 6C50, 6C51, 6C5Y, QE21, QE22.
