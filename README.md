# MYTH Teknik Görev - Sketch Generation

## 📦 İndirdiğin Dosyalar

1. **sketch_diffusion_solution.ipynb** - Ana çalışma notebook'u (TAM ÇALIŞIR!)
2. **KURULUM_REHBERI.md** - Detaylı kurulum adımları
3. **TEK_KOMUT_KURULUM.md** - Copy-paste tek komut kurulum
4. **QUICK_START.md** - Hızlı başlangıç rehberi
5. **CALISMA_PLANI.md** - 7 günlük detaylı plan
6. **setup_project.sh** - Otomatik kurulum script'i

---

## 🚀 Hemen Başlamak İçin

### 1. Proje Klasörü Oluştur
```bash
cd ~/Desktop
mkdir myth_sketch_diffusion
cd myth_sketch_diffusion
```

### 2. Dosyaları Taşı
- İndirdiğin **tüm dosyaları** bu klasöre taşı
- **subset.zip**'i de (assignment'tan) buraya koy

### 3. Kurulumu Yap
İki seçenek var:

**Seçenek A: Otomatik (önerilen)**
```bash
bash setup_project.sh
```

**Seçenek B: Manuel**
`TEK_KOMUT_KURULUM.md` dosyasını aç, komutları kopyala-yapıştır.

### 4. Notebook'u Başlat
```bash
conda activate sketch
jupyter notebook sketch_diffusion_solution.ipynb
```

---

## 📚 Hangi Dosyayı Okumalısın?

### Şimdi Oku:
1. **TEK_KOMUT_KURULUM.md** → Kurulumu yap
2. **QUICK_START.md** → İlk adımlar

### Sonra Oku:
3. **CALISMA_PLANI.md** → 7 günlük detaylı plan
4. **KURULUM_REHBERI.md** → Sorun çıkarsa buraya bak

### Çalıştır:
5. **sketch_diffusion_solution.ipynb** → Ana kod

---

## ⏱️ Tahmini Süreler

| Adım | Süre |
|------|------|
| Kurulum | 15-20 dakika |
| Veri indirme | 10-15 dakika |
| Cat modeli eğitimi | 4-6 saat |
| Bus modeli eğitimi | 4-6 saat |
| Rabbit modeli eğitimi | 4-6 saat |
| Visualization | 1-2 saat |
| FID/KID hesaplama | 1-2 saat |
| **TOPLAM** | **~20 saat** |

💡 **Quick test** için: Her modeli 10 epoch eğit (toplam ~3 saat)

---

## 🎯 Ne Yapacaksın?

1. ✅ Environment kur (conda + pip)
2. ✅ Dataset indir (cat, bus, rabbit NDJSON'ları)
3. ✅ 3 model eğit (cat, bus, rabbit için ayrı ayrı)
4. ✅ GIF'ler oluştur (stroke-by-stroke animasyon)
5. ✅ FID/KID hesapla (quantitative evaluation)
6. ✅ Notebook'u temizle ve GitHub'a yükle

---

## 🔥 Kritik Noktalar

- **GPU ŞART**: CPU'da çok yavaş (100x). Google Colab/Kaggle kullan.
- **Checkpoint kaydet**: Her 10 epoch'ta kaydet (elektrik giderse boşa gitmesin)
- **İlk test et**: 5-10 epoch ile sistemi test et, sonra full eğitim
- **Deadline**: 1 hafta, model eğitimi en uzun kısım (3-4 gün)

---

## 🐛 Sorun Çıkarsa

1. **KURULUM_REHBERI.md** → "Sorun Giderme" bölümüne bak
2. **Google'da ara**: Hata mesajını kopyala, google'a yapıştır
3. **Claude'a sor**: Ben buradayım! 

---

## ✅ Başarı Kriterleri

Assignment'ı tamamlamak için:

- [ ] 3 kategori için eğitilmiş model
- [ ] 3 FID score (cat, bus, rabbit)
- [ ] 3 KID score (cat, bus, rabbit)
- [ ] 9 adet GIF (3 kategori × 3 sample)
- [ ] Generated sample visualizations
- [ ] Tam çalışır Jupyter notebook
- [ ] GitHub repository

---

## 📊 Beklenen Sonuçlar

- **FID**: 50-150 arası (sketch domain'de normal)
- **KID**: 0.01-0.05 arası
- **GIF'ler**: Smooth stroke-by-stroke drawing
- **Görsel kalite**: Tanınabilir objeler

---

## 💪 Şimdi Ne Yapmalısın?

### Bugün (0-2 saat):
```bash
# 1. Proje klasörü oluştur
cd ~/Desktop && mkdir myth_sketch_diffusion && cd myth_sketch_diffusion

# 2. Dosyaları taşı (tüm .md, .ipynb, .sh dosyaları + subset.zip)

# 3. Kurulum yap
bash setup_project.sh

# 4. İlk test
jupyter notebook sketch_diffusion_solution.ipynb
# → İlk 5-10 cell'i çalıştır, sistemi test et
```

### Yarın (5-8 saat):
```bash
# 5. Cat modelini eğit (50 epoch)
# Notebook'ta cat training cell'ini çalıştır
```

### Öbür Gün (5-8 saat):
```bash
# 6. Bus ve Rabbit modellerini eğit
```

### 4. Gün (3-4 saat):
```bash
# 7. Generation + GIF'ler + FID/KID
```

### 5. Gün (2-3 saat):
```bash
# 8. Notebook'u temizle, GitHub'a yükle
```

---

## 🎓 Son Tavsiyeler

1. **Erken başla**: Model eğitimi UZUN sürer
2. **GPU kullan**: Colab/Kaggle ücretsiz
3. **Checkpoint kaydet**: 10 epoch'ta bir kaydet
4. **Test et önce**: 5 epoch quick test yap
5. **Dokümante et**: Ne yaptığını açıkla

---

## 📞 Yardım

Takıldığın yerde:
1. İlgili .md dosyasını oku
2. Google'da ara  
3. Claude'a sor (ben buradayım!)

---

**Hazırsın! Şimdi TEK_KOMUT_KURULUM.md'yi aç ve başla! 🚀**

Good luck! 💪
