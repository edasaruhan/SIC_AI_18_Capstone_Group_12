import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Terminaldeki gereksiz uyarıları gizle
warnings.filterwarnings('ignore')

print("Veriler okunuyor, lütfen bekleyin...")

# 1. Dosya yollarını mevcut klasör yapısına göre belirle
pinfo_yolu = "mobile_phone_use/pinfo.csv"
kullanici_verisi_yolu = "mobile_phone_use/data/u1c677c.csv" # u1c677c.csv'nin data klasöründe olduğunu varsayıyoruz

# 2. Verileri yükle
pinfo_df = pd.read_csv(pinfo_yolu)
u_data = pd.read_csv(kullanici_verisi_yolu, low_memory=False)

# Zaman damgasını datetime'a çevir (GMT değil, yerel saat olan ts sütunu kullanılıyor)
u_data['ts'] = pd.to_datetime(u_data['ts'])

# 3. Sadece Ekran (Screen) ve Uygulama (App) verilerini filtrele
screen_data = u_data[u_data['sensor_id'] == 'Screen'].copy()
app_data = u_data[u_data['sensor_id'] == 'App'].copy()

# 4. Günlük bazda kilit açma (Unlocked) sayısını hesapla
unlocks = screen_data[screen_data['Screen_Value'] == 'Unlocked']
daily_unlocks = unlocks.groupby(unlocks['ts'].dt.date).size()

# 5. Gece kullanımı (00:00 - 05:00 arası kilit açma)
night_unlocks = unlocks[(unlocks['ts'].dt.hour >= 0) & (unlocks['ts'].dt.hour < 5)]
daily_night_unlocks = night_unlocks.groupby(night_unlocks['ts'].dt.date).size()

# 6. Uygulama değiştirme (App switches) sayısını hesapla
daily_app_switches = app_data.groupby(app_data['ts'].dt.date).size()

# 7. Sonuçları bir DataFrame'de birleştir
summary_df = pd.DataFrame({
    'Toplam_Kilit_Acma': daily_unlocks,
    'Gece_Kilit_Acma': daily_night_unlocks,
    'Uygulama_Gecis_Sayisi': daily_app_switches
}).fillna(0).astype(int)

print("\n--- Örnek Katılımcı (u1c677c) Genel Ortalamalar ---")
print(summary_df.mean().round(2))

# 8. UNHOOK için Grafik Çizdirme
plt.figure(figsize=(10, 5))
plt.plot(summary_df.index, summary_df['Toplam_Kilit_Acma'], marker='o', label='Günlük Toplam Kilit Açma', color='#2ca02c')
plt.plot(summary_df.index, summary_df['Gece_Kilit_Acma'], marker='o', label='Gece (00:00-05:00) Kilit Açma', color='#d62728')
plt.title('Katılımcı (u1c677c) Günlük ve Gece Kilit Açma Sıklığı')
plt.xlabel('Tarih')
plt.ylabel('Kilit Açma Sayısı')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Grafiği kaydet ve ekranda göster
plt.savefig('kilit_acma_grafigi.png')
print("\nGrafik 'kilit_acma_grafigi.png' olarak kaydedildi.")
plt.show()